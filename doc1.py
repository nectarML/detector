import streamlit as st
import pandas as pd
import joblib
# Загрузка готовой модели
model = joblib.load('svc.pkl')
# Создание заголовка и описания приложения
st.title('Определитель талантов учеников')
st.write('Введите данные для получения предсказания')
# Создание формы для ввода данных
user_input = []
st.caption('Первый год')
user_input.append(st.number_input('Математика', value=0.0, key=0));
user_input.append(st.number_input('Русский язык', value=0.0, key=1));
user_input.append(st.number_input('Биология', value=0.0, key=2));
user_input.append(st.number_input('Английский язык', value=0.0, key=3));
user_input.append(st.number_input('География', value=0.0, key=4));
user_input.append(st.number_input('Физика', value=0.0, key=5));
user_input.append(st.number_input('Химия', value=0.0, key=6));
user_input.append(st.number_input('Обществознание', value=0.0, key=7));
user_input.append(st.number_input('Информатика', value=0.0, key=8));
st.caption('Второй год')
user_input.append(st.number_input('Математика', value=0.0, key=9));
user_input.append(st.number_input('Русский язык', value=0.0, key=10));
user_input.append(st.number_input('Биология', value=0.0, key=11));
user_input.append(st.number_input('Английский язык', value=0.0, key=12));
user_input.append(st.number_input('География', value=0.0, key=13));
user_input.append(st.number_input('Физика', value=0.0, key=14));
user_input.append(st.number_input('Химия', value=0.0, key=15));
user_input.append(st.number_input('Обществознание', value=0.0, key=16));
user_input.append(st.number_input('Информатика', value=0.0, key=17));
# Преобразование пользовательского ввода в DataFrame
df1 = pd.DataFrame([user_input])
print(df1.columns.tolist())
# Добавление столбца 'omathinfo'
# Добавление столбца 'omathinfo' с условием
df1['sum1'] = df1[0] + df1[1] + df1[2] + df1[3]+df1[4]+df1[5] + df1[6] + df1[7] + df1[8]
df1['sum2'] = df1[9] + df1[10] + df1[11] + df1[12]+df1[13]+df1[14] + df1[15] + df1[16] + df1[17]
print(df1['sum1'])
df1['mathinfo1'] = ((df1[0] + df1[8]) / df1['sum1']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['mathinfo2'] = ((df1[9] + df1[17]) / df1['sum2']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['omathinfo'] = (df1['mathinfo1']+ df1['mathinfo2'])
df1 = df1.drop(['mathinfo1','mathinfo2'], axis=1)
df1['russoc1'] = ((df1[1] + df1[7]) / df1['sum1']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['russoc2'] = ((df1[10] + df1[16]) / df1['sum2']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['orussoc'] = (df1['russoc1']+ df1['russoc2'])
df1 = df1.drop(['russoc1','russoc2'], axis=1)
df1['fizmath1'] = ((df1[0] + df1[5]) / df1['sum1']).apply(lambda x: 1 if x > 0.243 else 0)
df1['fizmath2'] = ((df1[9] + df1[14]) / df1['sum2']).apply(lambda x: 1 if x > 0.243 else 0)
df1['ofizmath'] = (df1['fizmath1']+ df1['fizmath2'])
df1 = df1.drop(['fizmath1','fizmath2'], axis=1)
df1['biochem1'] = ((df1[2] + df1[6]) / df1['sum1']).apply(lambda x: 1 if x > 0.243 else 0)
df1['biochem2'] = ((df1[11] + df1[15]) / df1['sum2']).apply(lambda x: 1 if x > 0.243 else 0)
df1['obiochem'] = (df1['biochem1']+ df1['biochem2'])
df1 = df1.drop(['biochem1','biochem2'], axis=1)
df1['geo11'] = (df1[4] / df1['sum1']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['geo21'] = (df1[13] / df1['sum2']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['ogeo'] = (df1['geo11']+ df1['geo21'])
df1 = df1.drop(['geo11','geo21'], axis=1)
df1['chem11'] = (df1[6] / df1['sum1']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['chem21'] = (df1[15] / df1['sum2']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['ochem'] = (df1['chem11']+ df1['chem21'])
df1 = df1.drop(['chem11','chem21'], axis=1)
df1['eng11'] = (df1[3] / df1['sum1']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['eng21'] = (df1[12] / df1['sum2']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['oeng'] = (df1['eng11']+ df1['eng21'])
df1 = df1.drop(['eng11','eng21'], axis=1)
df1 = df1.drop(['sum1','sum2'],axis=1)
df1 = df1.rename(columns={0: 'math1'})
df1 = df1.rename(columns={1: 'rus1'})
df1 = df1.rename(columns={2: 'bio1'})
df1 = df1.rename(columns={3: 'eng1'})
df1 = df1.rename(columns={4: 'geo1'})
df1 = df1.rename(columns={5: 'fiz1'})
df1 = df1.rename(columns={6: 'chem1'})
df1 = df1.rename(columns={7: 'soc1'})
df1 = df1.rename(columns={8: 'info1'})
df1 = df1.rename(columns={9: 'math2'})
df1 = df1.rename(columns={10: 'rus2'})
df1 = df1.rename(columns={11: 'bio2'})
df1 = df1.rename(columns={12: 'eng2'})
df1 = df1.rename(columns={13: 'geo2'})
df1 = df1.rename(columns={14: 'fiz2'})
df1 = df1.rename(columns={15: 'chem2'})
df1 = df1.rename(columns={16: 'soc2'})
df1 = df1.rename(columns={17: 'info2'})
# Заменяем "omathinfo" на "математика"

df1.columns = df1.columns.astype(str)
# Отображение пользовательского ввода
# Предсказание с помощью загруженной модели
if st.button('Получить предсказание'):
    prediction = model.predict(df1)
    if prediction > 6:
        st.write("Явная склонность ученика к направлению")
    elif 3 <= prediction <= 6:
        st.write("Вероятно ученик имеет склонность")
    else:
        st.write("Ученик не имеет особенных склонностей ни к одному из предметов, в целом учится одинаково")
    if (df1['math1'] + df1['math2']).sum() < 7:
        st.write("Обнаружены систематические проблемы в учебе: Математика")
    if (df1['rus1'] + df1['rus2']).sum() < 7:
        st.write("Обнаружены систематические проблемы в учебе: Русский")
    if (df1['fiz1'] + df1['fiz2']).sum() < 7:
        st.write("Обнаружены систематические проблемы в учебе: Физика")
    if (df1['bio1'] + df1['bio2']).sum() < 7:
        st.write("Обнаружены систематические проблемы в учебе: Биология")
    if (df1['soc1'] + df1['soc2']).sum() < 7:
        st.write("Обнаружены систематические проблемы в учебе: Общество")
    if (df1['chem1'] + df1['chem2']).sum() < 7:
        st.write("Обнаружены систематические проблемы в учебе: Химия")
    if (df1['info1'] + df1['info2']).sum() < 7:
        st.write("Обнаружены систематические проблемы в учебе: Информатика") 
    # Заменяем "omathinfo" на "математика"
    # Создаем новый датафрейм с двумя колонками 'name' и 'grade'
    names = list(df1.columns[18:25])  # Получаем список названий колонок
    grades = df1.iloc[:, 18:25].values.flatten().tolist()  # Получаем список значений колонок с 18 по 24
    new_df = pd.DataFrame({'name': names, 'grade': grades})
    # Заменяем "omathinfo" на "математика"
    new_df['name'] = new_df['name'].replace('omathinfo', 'математика и информатика')
    new_df['name'] = new_df['name'].replace('orussoc', 'русский и обществознание')
    new_df['name'] = new_df['name'].replace('ofizmath', 'математика и физика')
    new_df['name'] = new_df['name'].replace('obiochem', 'биология и химия')
    new_df['name'] = new_df['name'].replace('ogeo', 'география')
    new_df['name'] = new_df['name'].replace('ochem', 'химия')
    new_df['name'] = new_df['name'].replace('oeng', 'английский')
    # plot multiple columns such as population and year from dataframe
    st.line_chart(new_df.set_index('name'))
    st.write('По оси Y отмечены общий потенциал, чем больше значение, тем сильнее выражена направленность ученика конкретно к этому предмету, значение 1 означет, что ученику легко даются данные предметы')
    st.write("По оси X отмечены несколько основных направлений")
