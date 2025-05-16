# -*- coding: utf-8 -*-
"""
Частотный анализ продвигаемого сайта vs конкурентов + сравнение с корпусом + экспорт в Excel.
Добавлен лист «Comparison» с условным форматированием.
Зависимости:
    pip install requests beautifulsoup4 lxml nltk pymorphy2 pandas numpy tqdm openpyxl xlsxwriter
"""

import inspect, os, csv, requests, re, nltk, pymorphy2, random
import pandas as pd, numpy as np
from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm

# --- Патч для pymorphy2 на Python ≥3.11 ---
def _getargspec(func):
    spec = inspect.getfullargspec(func)
    return spec.args, spec.varargs, spec.varkw, spec.defaults
inspect.getargspec = _getargspec

# --- Стоп-слова NLTK ---
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def download_corpus_if_needed(script_dir: str):
    """Скачиваем HermitDave и сохраняем в russian_freq.csv"""
    hd_url = (
        'https://raw.githubusercontent.com/'
        'hermitdave/FrequencyWords/master/content/2016/ru/ru_full.txt'
    )
    print("⏳ Скачиваем HermitDave...")
    r = requests.get(hd_url, timeout=60)
    r.raise_for_status()
    lines = r.text.splitlines()
    out = os.path.join(script_dir, 'russian_freq.csv')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('word,freq_corpus\n')
        for L in lines:
            p = L.split()
            if len(p) >= 2:
                f.write(f"{p[0]},{p[1]}\n")
    print("✅ Сохранено →", out)

def load_corpus(path: str) -> pd.DataFrame:
    """Читаем CSV корпуса и считаем процентную частоту"""
    with open(path, 'r', encoding='utf-8-sig') as f:
        sample = f.read(2048)
        delim = csv.Sniffer().sniff(sample,
                                    delimiters=[',',';','\t','|']).delimiter
    reader = pd.read_csv(path, sep=delim, engine='python',
                         encoding='utf-8-sig',
                         usecols=lambda c: c.strip() in ('word','freq_corpus'),
                         skipinitialspace=True, chunksize=100_000)
    chunks, total = [], 0
    for chunk in tqdm(reader, desc="  чтение корпуса", unit="строк"):
        chunk.columns = chunk.columns.str.strip()
        chunks.append(chunk)
        total += chunk['freq_corpus'].sum()
    df = pd.concat(chunks, ignore_index=True)
    df['percent_corpus'] = df['freq_corpus'] / total * 100
    return df.rename(columns={'word':'Лемма'})[['Лемма','percent_corpus']]

def get_page_text(session: requests.Session,
                  url: str,
                  proxy: str = None) -> str:
    """Скачиваем страницу, удаляем служебные теги и возвращаем чистый текст"""
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/114.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9',
        'Connection': 'keep-alive',
        'Referer': url,
    }
    kwargs = {}
    if proxy:
        kwargs['proxies'] = {'http': proxy, 'https': proxy}
    resp = session.get(url, headers=headers, timeout=30, **kwargs)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'lxml')
    for tag in soup(['script','style','nav','header','footer','aside']):
        tag.decompose()
    return soup.body.get_text(separator=' ', strip=True) if soup.body else soup.get_text()

def tokenize_morph(text: str) -> list[dict]:
    """Разбиваем на токены, нормализуем pymorphy2 и фильтруем стоп-слова"""
    tokens = re.findall(r'\b[а-яё]+\b', text.lower())
    out = []
    for w in tqdm(tokens, desc="  морфо-токены", unit="токен"):
        if w in STOP_RU or len(w) < 3:
            continue
        p = MORPH.parse(w)[0]
        out.append({'word': w,
                    'lemma': p.normal_form,
                    'pos': p.tag.POS or ''})
    return out

def build_target_freq(morph_data: list[dict],
                      top_n: int = 100) -> pd.DataFrame:
    """Считаем абсолютные и относительные частоты для топ-N лемм"""
    total = len(morph_data) or 1
    cnt = Counter(item['word'] for item in morph_data)
    rows = []
    for word, freq in cnt.most_common(top_n):
        info = next(item for item in morph_data if item['word'] == word)
        rows.append({
            'Слово': word,
            'Лемма': info['lemma'],
            'Часть речи': info['pos'],
            'Частота': freq,
            '%_target': round(freq / total * 100, 3)
        })
    return pd.DataFrame(rows)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ========== Инициализация MorphAnalyzer и стоп-слов ==========
    MORPH = pymorphy2.MorphAnalyzer()
    STOP_RU = set(stopwords.words('russian'))

    # 1) URL продвигаемого сайта
    main_url = input("Введите URL продвигаемого сайта: ").strip()

    # 2) URL конкурентов
    raw = input("Введите URL конкурентов через запятую (или Enter, если без): ")
    comp_urls = [u.strip() for u in raw.split(',') if u.strip()]

    # 3) Использовать прокси?
    use_proxy = input("Использовать прокси? (да/нет): ").strip().lower() in ('да','д','yes','y')
    proxies: list[str] = []
    if use_proxy:
        print("Введите прокси по одному (host:port или http://user:pass@host:port). Пустая строка — конец ввода.")
        while True:
            p = input("Прокси: ").strip()
            if not p:
                break
            proxies.append(p)
        if not proxies:
            print("⚠️ Прокси не введены — продолжим без прокси.")
            use_proxy = False

    # 4) Скачать HermitDave?
    if input("Скачать корпус HermitDave? (да/нет): ").strip().lower() in ('да','д','yes','y'):
        download_corpus_if_needed(script_dir)

    # 5) Выбор CSV-корпуса в папке со скриптом
    csvs = [f for f in os.listdir(script_dir) if f.lower().endswith('.csv')]
    print("\nДоступные CSV корпуса:")
    for i, fn in enumerate(csvs, 1):
        print(f"  {i}. {fn}")
    idx = int(input(f"Выберите корпус (1–{len(csvs)}): ").strip()) - 1
    corpus_df = load_corpus(os.path.join(script_dir, csvs[idx]))

    # 6) Top-N
    try:
        top_n = int(input("Сколько топ-слов брать? (по умолчанию 100): ") or 100)
    except ValueError:
        top_n = 100

    # 7) Подготовка сессии
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (анализатор)'})

    # 8) Анализ главного сайта
    print(f"\n🔍 Анализ продвигаемого сайта: {main_url}")
    proxy = random.choice(proxies) if use_proxy else None
    main_text = get_page_text(session, main_url, proxy)
    main_morph = tokenize_morph(main_text)
    df_main = build_target_freq(main_morph, top_n)
    df_main['URL'] = main_url

    # 9) Анализ конкурентов
    results_comp: dict[str, pd.DataFrame] = {}
    for url in comp_urls:
        print(f"\n🔍 Анализ конкурента: {url}")
        proxy = random.choice(proxies) if use_proxy else None
        text = get_page_text(session, url, proxy)
        morph = tokenize_morph(text)
        df = build_target_freq(morph, top_n)
        df['URL'] = url
        results_comp[url] = df

    # 10) Объединяем и считаем относительные показатели
    all_df = pd.concat([df_main] + list(results_comp.values()), ignore_index=True)
    merged = all_df.merge(corpus_df, how='left', on='Лемма')
    mz = merged['percent_corpus'][merged['percent_corpus']>0].min()
    merged['percent_corpus'] = merged['percent_corpus'].fillna(mz)
    merged['%_corpus']  = merged['percent_corpus'].round(3)
    merged['Ratio']     = (merged['%_target'] / merged['%_corpus']).round(3)
    merged['Log(Ratio)']= np.log(merged['Ratio']).round(3)
    comp_merged = merged[merged['URL'].isin(comp_urls)]

    # 11) Общие и уникальные леммы
    all_urls = [main_url] + comp_urls
    n_pages = len(all_urls)

    common = (
        merged
        .groupby('Лемма')['URL']
        .nunique()
        .reset_index(name='pages')
    )
    common = common[common['pages'] == n_pages].drop(columns='pages')

    unique: dict[str, pd.DataFrame] = {}
    for url in all_urls:
        df_url = merged[merged['URL'] == url]
        others = set(merged.loc[merged['URL'] != url, 'Лемма'])
        unique[url] = df_url[~df_url['Лемма'].isin(others)].copy()

    # === 12) Средние по конкурентам и Comparison-DataFrame ===
    comp_stats = comp_merged.groupby('Лемма').agg({
        'Частота': 'mean',
        '%_target': 'mean',
        'Ratio': 'mean'
    }).rename(columns={
        'Частота': 'Частота_comp_avg',
        '%_target': '%_target_comp_avg',
        'Ratio': 'Ratio_comp_avg'
    }).reset_index()

    # Округление корректным способом
    comp_stats = comp_stats.round({
        'Частота_comp_avg': 2,
        '%_target_comp_avg': 3,
        'Ratio_comp_avg': 3
    })


    # === ВСТАВКА 1: Добавляем колонку "Что делать" ===
    def decide_action(row):
        if row['%_target'] > row['%_target_comp_avg']:
            return 'Уменьшить'
        elif row['%_target'] < row['%_target_comp_avg']:
            return 'Добавить'
        else:
            return 'Ничего не делать'


    comp_df = df_main.merge(comp_stats, on='Лемма', how='left')
    comp_df['Что делать'] = comp_df.apply(decide_action, axis=1)
    # === Конец вставки 1 ===

    # === 13) Экспорт всех листов в Excel ===
    out_xlsx = os.path.join(script_dir, 'multi_page_analysis.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
        merged.to_excel(writer, sheet_name='All', index=False)
        common.to_excel(writer, sheet_name='Common', index=False)
        # … листы Unique_… …
        comp_df.to_excel(writer, sheet_name='Comparison', index=False)

        # === ВСТАВКА 2: Записываем отдельный лист "Что делать" ===
        comp_df[
            ['Слово', 'Лемма', '%_target', '%_target_comp_avg', 'Ratio_comp_avg', 'Что делать']
        ].to_excel(
            writer,
            sheet_name='Что делать',
            index=False
        )
        # === Конец вставки 2 ===

        # Условное форматирование для листа Comparison
        workbook = writer.book
        worksheet = writer.sheets['Comparison']
    # 13) Экспорт всех листов в Excel
    out_xlsx = os.path.join(script_dir, 'multi_page_analysis.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
        merged.to_excel(writer, sheet_name='All', index=False)
        common.to_excel(writer, sheet_name='Common', index=False)

        prefix = 'Unique_'
        max_len = 31 - len(prefix)
        for url, df_u in unique.items():
            raw = url.replace('https://','').replace('http://','').replace('/','_')
            sheet = prefix + raw[:max_len]
            df_u.to_excel(writer, sheet_name=sheet, index=False)

        comp_df.to_excel(writer, sheet_name='Comparison', index=False)

        # Условное форматирование для Comparison
        workbook  = writer.book
        worksheet = writer.sheets['Comparison']
        fmt_g = workbook.add_format({'bg_color':'#C6EFCE','font_color':'#006100'})
        fmt_r = workbook.add_format({'bg_color':'#FFC7CE','font_color':'#9C0006'})
        fmt_y = workbook.add_format({'bg_color':'#FFEB9C','font_color':'#9C6500'})
        nrows = len(comp_df) + 1

        # D vs E (Частота)
        worksheet.conditional_format(f'D2:D{nrows}', {
            'type':'formula','criteria':'=D2>E2','format':fmt_g})
        worksheet.conditional_format(f'D2:D{nrows}', {
            'type':'formula','criteria':'=D2<E2','format':fmt_r})
        worksheet.conditional_format(f'D2:D{nrows}', {
            'type':'formula','criteria':'=D2=E2','format':fmt_y})
        # F vs G (%_target)
        worksheet.conditional_format(f'F2:F{nrows}', {
            'type':'formula','criteria':'=F2>G2','format':fmt_g})
        worksheet.conditional_format(f'F2:F{nrows}', {
            'type':'formula','criteria':'=F2<G2','format':fmt_r})
        worksheet.conditional_format(f'F2:F{nrows}', {
            'type':'formula','criteria':'=F2=G2','format':fmt_y})

    print(f"\n✅ Анализ завершён. Результаты в {out_xlsx}")
