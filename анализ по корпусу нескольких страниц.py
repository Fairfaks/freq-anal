# -*- coding: utf-8 -*-
"""
Частотный анализ продвигаемого сайта vs конкурентов + сравнение с корпусом + экспорт в Excel.
Добавлен лист «Comparison» с условным форматированием.
Зависимости:
    pip install requests beautifulsoup4 lxml nltk pymorphy2 pandas numpy tqdm openpyxl xlsxwriter
"""

import inspect, os, csv, requests, re, nltk, pymorphy2
import pandas as pd, numpy as np
from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm

# --- Патч для pymorphy2 + Python 3.11+ ---
def _getargspec(func):
    spec = inspect.getfullargspec(func)
    return spec.args, spec.varargs, spec.varkw, spec.defaults
inspect.getargspec = _getargspec

# --- Стоп-слова NLTK ---
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def get_page_text(url: str) -> str:
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/114.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9',
        'Connection': 'keep-alive','Referer': url,
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'lxml')
    for tag in soup(['script','style','nav','header','footer','aside']):
        tag.decompose()
    return soup.body.get_text(separator=' ', strip=True) if soup.body else soup.get_text()

def tokenize_morph(text: str):
    tokens = re.findall(r'\b[а-яё]+\b', text.lower())
    morph = pymorphy2.MorphAnalyzer()
    stop_ru = set(stopwords.words('russian'))
    out = []
    for w in tqdm(tokens, desc="  морфо-токены", unit="токен"):
        if w in stop_ru or len(w)<3: continue
        p = morph.parse(w)[0]
        out.append({'word':w,'lemma':p.normal_form,'pos':p.tag.POS or ''})
    return out

def build_target_freq(morph_data, top_n=100):
    total = len(morph_data)
    cnt = Counter(item['word'] for item in morph_data)
    rows = []
    for word,freq in cnt.most_common(top_n):
        info = next(item for item in morph_data if item['word']==word)
        rows.append({
            'Слово':word,
            'Лемма':info['lemma'],
            'Часть речи':info['pos'],
            'Частота':freq,
            '%_target':round(freq/total*100,3)
        })
    return pd.DataFrame(rows)

def load_corpus(path: str) -> pd.DataFrame:
    with open(path,'r',encoding='utf-8-sig') as f:
        sample = f.read(2048)
        delim = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|']).delimiter
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
    df['percent_corpus'] = df['freq_corpus']/total*100
    return df.rename(columns={'word':'Лемма'})[['Лемма','percent_corpus']]

if __name__=='__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) Ввод URL продвигаемого сайта
    main_url = input("Введите URL продвигаемого сайта: ").strip()

    # 2) Ввод URL конкурентов
    raw = input("Введите URL конкурентов через запятую: ").split(',')
    comp_urls = [u.strip() for u in raw if u.strip()]

    # 3) Опционально скачиваем корпус HermitDave
    if input("Скачать корпус HermitDave? (Да/Нет): ").strip().lower() in ('да','д','yes','y'):
        hd_url = 'https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/ru/ru_full.txt'
        print("⏳ Скачиваем HermitDave...")
        r = requests.get(hd_url, timeout=60); r.raise_for_status()
        lines = r.text.splitlines()
        out = os.path.join(script_dir,'russian_freq.csv')
        with open(out,'w',encoding='utf-8') as f:
            f.write('word,freq_corpus\n')
            for L in lines:
                p = L.split()
                if len(p)>=2: f.write(f"{p[0]},{p[1]}\n")
        print("✅ Сохранено →", out)

    # 4) Выбор корпуса
    csvs = [f for f in os.listdir(script_dir) if f.lower().endswith('.csv')]
    print("\nДоступные CSV корпуса:")
    for i,fn in enumerate(csvs,1): print(f"  {i}. {fn}")
    idx = int(input(f"Выберите корпус (1–{len(csvs)}): ").strip())-1
    corpus_df = load_corpus(os.path.join(script_dir, csvs[idx]))

    # 5) Анализ продвигаемого сайта
    print(f"\n🔍 Анализ продвигаемого сайта: {main_url}")
    main_text = get_page_text(main_url)
    main_morph = tokenize_morph(main_text)
    df_main = build_target_freq(main_morph, top_n=100)
    df_main['URL'] = main_url

    # 6) Анализ конкурентов
    results_comp = {}
    for url in comp_urls:
        print(f"\n🔍 Анализ конкурента: {url}")
        text = get_page_text(url)
        morph = tokenize_morph(text)
        df = build_target_freq(morph, top_n=100)
        df['URL'] = url
        results_comp[url] = df

    # 7) Объединяем все в один DataFrame
    all_df = pd.concat([df_main] + list(results_comp.values()), ignore_index=True)
    merged = all_df.merge(corpus_df, how='left', on='Лемма')
    mz = merged['percent_corpus'][merged['percent_corpus']>0].min()
    merged['percent_corpus'] = merged['percent_corpus'].fillna(mz)
    merged['%_corpus']  = merged['percent_corpus'].round(3)
    merged['Ratio']     = (merged['%_target']/merged['%_corpus']).round(3)
    merged['Log(Ratio)']= np.log(merged['Ratio']).round(3)

    # 8) Общие и уникальные леммы
    all_urls = [main_url] + comp_urls
    n_pages = len(all_urls)

    # 8.1) Общие леммы
    common = (
        merged
        .groupby('Лемма')['URL']
        .nunique()
        .reset_index(name='pages')
    )
    common = common[common['pages'] == n_pages].drop(columns='pages')

    # 8.2) Уникальные леммы
    unique = {}
    for url in all_urls:
        df_url = merged[merged['URL'] == url]
        others = set(merged.loc[merged['URL'] != url, 'Лемма'])
        unique[url] = df_url[~df_url['Лемма'].isin(others)].copy()

    # 9) Средние по конкурентам: Абсолютная, Относительная и Ratio
    comp_merged = merged[merged['URL'].isin(comp_urls)]
    comp_stats = comp_merged.groupby('Лемма').agg({
        'Частота': 'mean',
        '%_target': 'mean',
        'Ratio': 'mean'
    }).rename(columns={
        'Частота': 'Частота_comp_avg',
        '%_target': '%_target_comp_avg',
        'Ratio': 'Ratio_comp_avg'
    }).reset_index()
    # округляем
    comp_stats['Частота_comp_avg'] = comp_stats['Частота_comp_avg'].round(2)
    comp_stats['%_target_comp_avg'] = comp_stats['%_target_comp_avg'].round(3)
    comp_stats['Ratio_comp_avg'] = comp_stats['Ratio_comp_avg'].round(3)

    # 10) Формируем Comparison-датафрейм
    comp_df = df_main.merge(comp_stats, on='Лемма', how='left')

    # 11) Экспорт всех листов в один файл
    out_xlsx = os.path.join(script_dir, 'multi_page_analysis.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
        # 11.1) Лист со всеми данными
        merged.to_excel(writer, sheet_name='All', index=False)

        # 11.2) Лист с общими леммами
        common.to_excel(writer, sheet_name='Common', index=False)

        # 11.3) Листы с уникальными леммами
        prefix = 'Unique_'
        max_raw = 31 - len(prefix)  # максимально допустимая длина «сырых» символов
        for url, df_u in unique.items():
            raw = url.replace('https://', '') \
                .replace('http://', '') \
                .replace('/', '_')
            sheet_name = prefix + raw[:max_raw]
            df_u.to_excel(writer, sheet_name=sheet_name, index=False)

        # 11.4) Лист Comparison
        comp_df.to_excel(writer, sheet_name='Comparison', index=False)

        # 11.5) Условное форматирование на листе Comparison
        workbook = writer.book
        worksheet = writer.sheets['Comparison']
        fmt_green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        fmt_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        fmt_yellow = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
        nrows = len(comp_df) + 1

        # подсветка абсолютных частот: D (main) vs E (comp_avg)
        worksheet.conditional_format(f'D2:D{nrows}', {
            'type': 'formula', 'criteria': '=D2>E2', 'format': fmt_green})
        worksheet.conditional_format(f'D2:D{nrows}', {
            'type': 'formula', 'criteria': '=D2<E2', 'format': fmt_red})
        worksheet.conditional_format(f'D2:D{nrows}', {
            'type': 'formula', 'criteria': '=D2=E2', 'format': fmt_yellow})

        # подсветка относительных частот: F (main) vs G (comp_avg)
        worksheet.conditional_format(f'F2:F{nrows}', {
            'type': 'formula', 'criteria': '=F2>G2', 'format': fmt_green})
        worksheet.conditional_format(f'F2:F{nrows}', {
            'type': 'formula', 'criteria': '=F2<G2', 'format': fmt_red})
        worksheet.conditional_format(f'F2:F{nrows}', {
            'type': 'formula', 'criteria': '=F2=G2', 'format': fmt_yellow})

    print(f"\n✅ Анализ завершён. Результаты в {out_xlsx}")
