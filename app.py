import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.data_preprocessing import get_data, remove_outliers
import plotly.express as px
import seaborn as sns

NUM_FEATURES = ['AGE', 'CHILD_TOTAL', 'EDUCATION', 'WORK_TIME',
                'PERSONAL_INCOME', 'CREDIT', 'LOANS_CLOSED']

def draw(data: pd.DataFrame):
    st.title('Пример данных:')
    st.dataframe(data.head())
    data_distribution(data)
    correlation(data)
    target(data)
    num_characteristics(data)


def data_distribution(data):
    st.title('Распределение признаков')
    age, gender, education, marital, dependants, address, job, income, loans, credit = st.tabs(
        tabs=['Возраст', 'Пол', 'Образование', 'Семейное положение',
              'Дети и иждивенцы', 'Адрес', 'Работа', 'Доход', 'Займы', 'Кредит'])
    with age:
        fig, ax = plt.subplots()
        age, count = data.AGE.value_counts().sort_index().index, data.AGE.value_counts().sort_index().values
        ax.bar(age, count)
        st.pyplot(fig)

    with gender:
        fig_gender, ax_gender = plt.subplots()
        gender_data = data.GENDER.replace({0: 'Женщина', 1: 'Мужчина'})
        gender, count = gender_data.value_counts().sort_index().index, gender_data.value_counts().sort_index().values
        ax_gender.bar(gender, count)
        st.pyplot(fig_gender)

    with education:
        fig_edu = px.pie(data.EDUCATION.value_counts().reset_index(), values='count', names='EDUCATION')
        st.plotly_chart(fig_edu, use_container_width=True)

    with marital:
        fig_edu = px.pie(data.MARITAL_STATUS.value_counts().reset_index(), values='count', names='MARITAL_STATUS')
        st.plotly_chart(fig_edu, use_container_width=True)

    with dependants:
        df = data.loc[:, ['CHILD_TOTAL', 'DEPENDANTS']].rename({'CHILD_TOTAL': 'CHILD'}, axis=1)
        df['TOTAL'] = df['CHILD']+df['DEPENDANTS']

        st.header('Дети')
        st.plotly_chart(px.bar(df.CHILD.value_counts().reset_index(), x='CHILD', y='count'))

        st.header('Иждивенцы')
        st.plotly_chart(px.bar(df.DEPENDANTS.value_counts().reset_index(), x='DEPENDANTS', y='count'))

        st.header('Всего')
        st.plotly_chart(px.bar(df.TOTAL.value_counts().reset_index(), x='TOTAL', y='count'))

    with address:
        st.header('Область регистрации')
        st.text('10 самых популярных областей')
        st.plotly_chart(px.pie(data.REG_ADDRESS_PROVINCE.value_counts().sort_values().reset_index()[:10],
                               values='count', names='REG_ADDRESS_PROVINCE'))

        st.header('Область проживания')
        st.text('10 самых популярных областей')
        st.plotly_chart(px.pie(data.FACT_ADDRESS_PROVINCE.value_counts().sort_values().reset_index()[:10],
                               values='count', names='FACT_ADDRESS_PROVINCE'))

        st.header('Совпадение области регистрации и проживания')
        match = data['FACT_ADDRESS_PROVINCE'] == data['REG_ADDRESS_PROVINCE']
        match.replace({True: 'Совпадает', False: 'Не совпадает'}, inplace=True)
        st.plotly_chart(px.bar(match.value_counts().reset_index(name='Количество'), x='index', y='Количество'))

    with job:
        st.header('Отрасль')
        st.plotly_chart(px.bar(data.GEN_INDUSTRY.value_counts().reset_index(), x='GEN_INDUSTRY', y='count'))

        st.header('Должность')
        st.plotly_chart(px.pie(data.GEN_TITLE.value_counts().reset_index(), names='GEN_TITLE', values='count'))

        st.header('Направление деятельности')
        st.plotly_chart(px.pie(data.JOB_DIR.value_counts().reset_index(), names='JOB_DIR', values='count'))

        st.header('Продолжительность работы (в месяцах)')
        work_time_filtered = data[data['WORK_TIME']/12 < data['AGE']]  # Отфильтруем ошибки (где стаж больше возраста)

        st.plotly_chart(px.histogram(work_time_filtered.WORK_TIME.value_counts().sort_index().reset_index(),
                                     x='WORK_TIME', y='count', nbins=150))
    with income:
        st.header('Доход семьи')
        fig_edu = px.pie(data.FAMILY_INCOME.value_counts().reset_index(), values='count', names='FAMILY_INCOME')
        st.plotly_chart(fig_edu, use_container_width=True)

        st.header('Личный доход')
        st.plotly_chart(px.histogram(data.PERSONAL_INCOME.value_counts().sort_index().reset_index(),
                                     x='PERSONAL_INCOME', y='count', nbins=100))

    with loans:
        st.header('Количество открытых займов')
        loans = data['LOANS_OPEN'] - data['LOANS_CLOSED']

        st.plotly_chart(px.histogram(loans.value_counts().sort_index().reset_index(),
                                     x='index', y='count', nbins=4))
    with credit:
        st.header('Сумма последнего кредита')
        st.plotly_chart(px.histogram(data.CREDIT.value_counts().sort_index().reset_index(),
                                     x='CREDIT', y='count'))

        st.header('Срок последнего кредита')
        st.plotly_chart(px.histogram(data.TERM.value_counts().sort_index().reset_index(),
                                     x='TERM', y='count'))

        st.header('Первоначальный взнос (в процентах)')
        percent = data[data['FST_PAYMENT'] < data['CREDIT']]
        percent = (percent['FST_PAYMENT']/percent['CREDIT'])*100
        st.plotly_chart(px.histogram(percent.value_counts().sort_index().reset_index(),
                                     x='index', y='count'))


def correlation(data):
    st.title('Корреляция признаков')
    users = data.loc[:, NUM_FEATURES]
    users.replace(
        {'Неполное среднее': 0, 'Среднее': 1, 'Среднее специальное': 2, 'Неоконченное высшее': 3, 'Высшее': 4,
         'Два и более высших образования': 5, 'Ученая степень': 6, 'не пенсионер': 1, 'пенсионер':0,
         'до 5000 руб.': 0, 'от 5000 до 10000 руб.': 1, 'от 10000 до 20000 руб.': 2, 'от 20000 до 50000 руб.': 3,
         'свыше 50000 руб.': 4}, inplace=True
    )

    plot = sns.heatmap(np.round(users.corr(), 3), annot=True)

    st.pyplot(plot.get_figure())

    st.text("""Видим сильную корреляцию между:
    - Количеством открытых и закрытых кредитов
    - Возрастом и статусом пенсионера
    И слабую между:
    - Возрастом и количеством детей
    - Образованием и доходом
    - Доходом и суммой последнего кредита""")


def target(data):
    st.title('Зависимость целевой переменной от признаков')

    data.replace(
        {'Неполное среднее': 0, 'Среднее': 1, 'Среднее специальное': 2, 'Неоконченное высшее': 3, 'Высшее': 4,
         'Два и более высших образования': 5, 'Ученая степень': 6, 'до 5000 руб.': 0, 'от 5000 до 10000 руб.': 1,
         'от 10000 до 20000 руб.': 2, 'от 20000 до 50000 руб.': 3, 'свыше 50000 руб.': 4}, inplace=True
    )
    data.TARGET.replace({0: 'Откликнулся', 1: 'Не откликнулся'})

    for feature in NUM_FEATURES:

        st.subheader(f'Зависимость таргета от {feature}')
        fig = plt.figure(figsize=(10, 4))
        sns.boxplot(remove_outliers(data, feature, 1), x='TARGET', y=feature)
        st.pyplot(fig)


def num_characteristics(data):
    st.title('Числовые характеристики')
    st.dataframe(data.loc[:, NUM_FEATURES].describe())


if __name__ == '__main__':
    draw(get_data('data'))