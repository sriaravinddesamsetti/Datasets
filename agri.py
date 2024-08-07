import streamlit as st
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import pyowm
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
owm = pyowm.OWM('11081b639d8ada3e97fc695bcf6ddb20')
from PIL import Image
import time
st.set_page_config(page_title = 'Agriculture', 
        layout='wide',page_icon=":mag_right:")
with st.sidebar:
    selected = option_menu("DashBoard", ["Home",'Weather','Crop Recommendation','Disease Detection','Fertilizer','Crop Production'], 
        icons=['house','cloud-sun','tree-fill','bug-fill','cart-plus-fill','clock-history'], menu_icon="cast", default_index=0,
        styles={
        "nav-link-selected": {"background-color": "green"},
    })
def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
if selected=='Home':
        st.markdown(f"<h1 style='text-align: center;font-size:60px;color:#33ccff;'>Agriculture</h1>", unsafe_allow_html=True)
        lottie_url = "https://assets8.lottiefiles.com/packages/lf20_CgexnTerux.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json,width=700,height=300)
if selected=='Weather':
        st.markdown(f"<h1 style='text-align: center; color:skyblue;'>Weather</h1>", unsafe_allow_html=True)
        id = st.text_input("Enter City")
        if len(id)==0:
            def load_lottieurl(url: str):
                    r = requests.get(url)
                    if r.status_code != 200:
                        return None
                    return r.json()
            lottie_url = "https://assets9.lottiefiles.com/temp/lf20_JA7Fsb.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json,width=700,height=300)
        else:
            try:
                mgr = owm.weather_manager()
                observation = mgr.weather_at_place(id)
                weather = observation.weather
                t1 = weather.temperature('celsius')['temp']
                h1 = weather.humidity
                w1 = weather.wind()
                p1=weather.pressure['press']
                num_weekdays = 5
                count_weekdays = 0
                weekday_names = []
                now = time.time()
                now1 = time.localtime()
                us_date = time.strftime("%m/%d/%Y", now1)
                while count_weekdays < num_weekdays:
                    now += 86400
                    local_time = time.localtime(now)
                    weekday = local_time.tm_wday
                    wn = time.strftime("%a", local_time)
                    if count_weekdays!=5:
                        count_weekdays += 1
                        weekday_names.append(time.strftime("%a", local_time))
                col1, col2,col3,col4= st.columns([2,8,5,3])
                col1.image('1.jpeg', width=75)
                with col4:
                    st.markdown(f"<h4>{'Weather'}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p>{us_date}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<h4 style='color:red;'>{t1}°C</h4>", unsafe_allow_html=True)
                    st.markdown(f"<hr/>",unsafe_allow_html=True)
                col1,col2,col3,col4= st.columns([5,8,5,5])
                col4.image('download (1).png', width=100)
                with col1:
                    st.markdown(f"<p>{'Humidity :  '}{h1}%</p>", unsafe_allow_html=True)
                with col1:
                    st.markdown(f"<p>{'Pressure :  '}{' '}{p1}hPa</p>", unsafe_allow_html=True)
                with col1:
                    st.markdown(f"<p>{'Wind Speed:  '}{w1['speed']}hPa</p>", unsafe_allow_html=True)  
                st.markdown(f"<hr/>",unsafe_allow_html=True)    
                col1, col2,col3,col4,col5= st.columns([4,4,4,4,4])
                forecaster = mgr.forecast_at_place(id, '3h', limit=40)

                c=0
                l=[]
                for weather in forecaster.forecast:
                    temperature = weather.temperature('celsius')['temp']
                    c+=1
                    if c==8 or c==16 or c==24 or c==32 or c==40:
                        l.append(temperature)
                with col1:
                    st.markdown(f"<h4 style='color:#EE82EE	';>{weekday_names[0]}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p>{l[0]}°C</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<h4 style='color:blue';>{weekday_names[1]}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p>{l[1]}°C</p>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<h4 style='color:green';>{weekday_names[2]}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p>{l[2]}°C</p>", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"<h4 style='color:orange';>{weekday_names[3]}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p>{l[3]}°C</p>", unsafe_allow_html=True)
                with col5:
                    st.markdown(f"<h4 style='color:red';>{weekday_names[4]}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p>{l[4]}°C</p>", unsafe_allow_html=True)
            except:
                def load_lottieurl(url: str):
                    r = requests.get(url)
                    if r.status_code != 200:
                        return None
                    return r.json()
                lottie_url = "https://assets10.lottiefiles.com/packages/lf20_biti0vdc.json"
                lottie_json = load_lottieurl(lottie_url)
                st_lottie(lottie_json,width=700,height=300)
if selected=='Crop Recommendation':
    st.markdown(f"<h1 style='text-align: center; color:red;'>Crop Recomendation</h1>", unsafe_allow_html=True)
    df=pd.read_csv("Crop_recommendation.csv")
    features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']
    labels = df['label']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
    RF = RandomForestClassifier(n_estimators=25, random_state=42)
    RF.fit(Xtrain,Ytrain)
    col1, col2,col3= st.columns([5,5,5])
    with col1:
        a=st.number_input('Enter N')
    with col2:
        b=st.number_input('Enter P')
    with col3:
        c1=st.number_input('Enter K')
    col1, col2,col3,col4= st.columns([5,5,5,5])
    with col1:
        d=st.number_input('Temperature °C')
    with col2:
        e=st.number_input('Humidity %')
    with col3:
        f=st.number_input('pH')
    with col4:
        g=st.number_input('Rainfall mm')
    data = np.array([[a,b,c1,d,e,f,g]])
    prediction = RF.predict(data)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        if prediction[0]=='apple':
            st.image("images/apple.jpg")
        if prediction[0]=='banana':
            st.image("images/banana.jpg")
        if prediction[0]=='blackgram':
            st.image("images/blackgram.jpg")
        if prediction[0]=='chickpea':
            st.image("images/chickpea.jpg")
        if prediction[0]=='coconut':
            st.image("images/coconut.jpg")
        if prediction[0]=='coffee':
            st.image("images/coffee.jpg")
        if prediction[0]=='cotton':
            st.image("images/cotton.jpg")
        if prediction[0]=='grapes':
            st.image("images/grapes.jpg")
        if prediction[0]=='jute':
            st.image("images/jute.jpg")
        if prediction[0]=='kidneybeans':
            st.image("images/kidneybeans.jpg")
        if prediction[0]=='lentil':
            st.image("images/lentil.jpg")
        if prediction[0]=='maize':
            st.image("images/maize.jpg")
        if prediction[0]=='mango':
            st.image("images/mango.jpg")
        if prediction[0]=='mothbeans':
            st.image("images/mothbeans.jpg")
        if prediction[0]=='mungbean':
            st.image("images/mungbeans.jpg")
        if prediction[0]=='muskmelon':
            st.image("images/muskmelon.jpg")
        if prediction[0]=='orange':
            st.image("images/orange.jpg")
        if prediction[0]=='papaya':
            st.image("images/papaya.jpg")
        if prediction[0]=='pomegranate':
            st.image("images/pomegranate.jpg")
        if prediction[0]=='pigeonpeas':
            st.image("images/pigeonpeas.jpg")
        if prediction[0]=='rice':
            st.image("images/rice.jpg")
        if prediction[0]=='watermelon':
            st.image("images/watermelon.jpg")
    with col3:
        st.write(' ')
if selected=='Fertilizer':
    st.markdown(f"<h1 style='text-align: center; color:blue;'>Fertilizer Prediction</h1>", unsafe_allow_html=True)
    data = pd.read_csv('Fertilizer Prediction.csv')
    data.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'},inplace=True)
    from sklearn.preprocessing import LabelEncoder
    encode_soil = LabelEncoder()
    data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)
    Soil_Type = pd.DataFrame(zip(encode_soil.classes_,encode_soil.transform(encode_soil.classes_)),columns=['Original','Encoded'])
    Soil_Type = Soil_Type.set_index('Original')
    encode_crop = LabelEncoder()
    data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)
    Crop_Type = pd.DataFrame(zip(encode_crop.classes_,encode_crop.transform(encode_crop.classes_)),columns=['Original','Encoded'])
    Crop_Type = Crop_Type.set_index('Original')
    encode_ferti = LabelEncoder()
    data.Fertilizer = encode_ferti.fit_transform(data.Fertilizer)
    Fertilizer = pd.DataFrame(zip(encode_ferti.classes_,encode_ferti.transform(encode_ferti.classes_)),columns=['Original','Encoded'])
    Fertilizer = Fertilizer.set_index('Original')
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data.drop('Fertilizer',axis=1),data.Fertilizer,test_size=0.2,random_state=1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    soil=['Black','Clayey','Loamy','Red','Sandy']
    crop=['Barley','Cotton','Ground Nuts','Maize','Millets','Oil seeds','Paddy','Pulses','Sugarcane','Tobacco','Wheat']
    fert=['10-26-26','14-35-14','17-17-17','20-20','28-28','DAP','Urea']
    rand = RandomForestClassifier(n_estimators=30,random_state=42)
    pred_rand = rand.fit(x_train,y_train).predict(x_test)
    col1, col2,col3= st.columns([5,5,5])
    with col1:
        a=st.number_input('Temperature °C')
    with col2:
        b=st.number_input('Humidity %')
    with col3:
        c=st.number_input('Moisture')
    col1,col2= st.columns([5,5])
    with col1:
        d=st.selectbox('Soil Type',('Black','Clayey','Loamy','Red','Sandy'))
    with col2:
        e=st.selectbox('Crop Type',('Barley','Cotton','Ground Nuts','Maize','Millets','Oil seeds','Paddy','Pulses','Sugarcane','Tobacco','Wheat'))
    col1, col2,col3= st.columns([5,5,5])
    with col1:
        f=st.number_input('Enter N')
    with col2:
        g=st.number_input('Enter P')
    with col3:
        h=st.number_input('Enter K')
    data = np.array([[a,b,c,soil.index(d),crop.index(e),f,g,h]])
    prediction = rand.predict(data)
    res=fert[prediction[0]]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        if res=='10-26-26':
            st.image("images/10-26-26.jpg")
        if res=='14-35-14':
            st.image("images/14-35-14.jpg")
        if res=='17-17-17':
            st.image("images/17-17-17.jpg")
        if res=='20-20':
            st.image("images/20-20.jpg")
        if res=='28-28':
            st.image("images/28-28.jpg")
        if res=='DAP':
            st.image("images/DAP.jpg")
        if res=='Urea':
            st.image("images/Urea.jpg")
    with col3:
        st.write(' ')
if selected=='Crop Production':
    st.markdown(f"<h1 style='text-align: center; color:blue;'>Crop Production</h1>", unsafe_allow_html=True)
    data=pd.read_csv("prod.csv")
    data.dropna(inplace=True)
    import seaborn as sns
    fig = plt.figure(figsize=(10, 4))
    st.markdown(f"<h4 style='color:black;'>State vs Crop Yield</h4>", unsafe_allow_html=True)
    input=st.selectbox("Select State",('Andaman and Nicobar Islands', 'Andhra Pradesh',
       'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh',
       'Chhattisgarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Delhi',
       'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
       'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
       'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
       'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan',
       'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh'))
    if input:
        filtered_df = data[(data['State'] == input) ]
        sns.barplot(x='State', y='Yield', data=filtered_df,hue=filtered_df['Crop'])
        st.pyplot(fig)
    st.markdown(f"<h4 style='color:black;'>District vs Crop Yield</h4>", unsafe_allow_html=True)
    inpu=st.selectbox("Select District",('NICOBARS', 'NORTH AND MIDDLE ANDAMAN', 'SOUTH ANDAMANS',
       'ANANTAPUR', 'EAST GODAVARI', 'KRISHNA', 'VIZIANAGARAM',
       'WEST GODAVARI', 'ADILABAD', 'CHITTOOR', 'GUNTUR', 'KADAPA',
       'KARIMNAGAR', 'KHAMMAM', 'KURNOOL', 'MAHBUBNAGAR', 'MEDAK',
       'NALGONDA', 'NIZAMABAD', 'PRAKASAM', 'RANGAREDDI', 'SPSR NELLORE',
       'SRIKAKULAM', 'VISAKHAPATANAM', 'WARANGAL', 'CHANGLANG',
       'DIBANG VALLEY', 'EAST KAMENG', 'EAST SIANG', 'KURUNG KUMEY',
       'LOHIT', 'LOWER DIBANG VALLEY', 'LOWER SUBANSIRI', 'PAPUM PARE',
       'TAWANG', 'TIRAP', 'UPPER SIANG', 'UPPER SUBANSIRI', 'WEST KAMENG',
       'WEST SIANG', 'BARPETA', 'BONGAIGAON', 'CACHAR', 'DARRANG',
       'DHEMAJI', 'DHUBRI', 'DIBRUGARH', 'DIMA HASAO', 'GOALPARA',
       'GOLAGHAT', 'HAILAKANDI', 'JORHAT', 'KAMRUP', 'KARBI ANGLONG',
       'KARIMGANJ', 'KOKRAJHAR', 'LAKHIMPUR', 'MARIGAON', 'NAGAON',
       'NALBARI', 'SIVASAGAR', 'SONITPUR', 'TINSUKIA', 'ARARIA', 'ARWAL',
       'AURANGABAD', 'BANKA', 'BEGUSARAI', 'BHAGALPUR', 'BHOJPUR',
       'BUXAR', 'DARBHANGA', 'GAYA', 'GOPALGANJ', 'JAMUI', 'JEHANABAD',
       'KAIMUR (BHABUA)', 'KATIHAR', 'KHAGARIA', 'KISHANGANJ',
       'LAKHISARAI', 'MADHEPURA', 'MADHUBANI', 'MUNGER', 'MUZAFFARPUR',
       'NALANDA', 'NAWADA', 'PASHCHIM CHAMPARAN', 'PATNA',
       'PURBI CHAMPARAN', 'PURNIA', 'ROHTAS', 'SAHARSA', 'SAMASTIPUR',
       'SARAN', 'SHEIKHPURA', 'SHEOHAR', 'SITAMARHI', 'SIWAN', 'SUPAUL',
       'VAISHALI', 'CHANDIGARH', 'BASTAR', 'BILASPUR', 'DANTEWADA',
       'DHAMTARI', 'DURG', 'JANJGIR-CHAMPA', 'JASHPUR', 'KABIRDHAM',
       'KANKER', 'KORBA', 'KOREA', 'MAHASAMUND', 'RAIGARH', 'RAIPUR',
       'RAJNANDGAON', 'SURGUJA', 'DADRA AND NAGAR HAVELI',
       'Daman and Diu', 'Delhi', 'Goa', 'AHMADABAD', 'AMRELI', 'ANAND',
       'BANAS KANTHA', 'BHARUCH', 'BHAVNAGAR', 'DANG', 'DOHAD',
       'GANDHINAGAR', 'JAMNAGAR', 'JUNAGADH', 'KACHCHH', 'KHEDA',
       'MAHESANA', 'NARMADA', 'NAVSARI', 'PANCH MAHALS', 'PATAN',
       'PORBANDAR', 'RAJKOT', 'SABAR KANTHA', 'SURAT', 'SURENDRANAGAR',
       'VADODARA', 'VALSAD', 'AMBALA', 'BHIWANI', 'FARIDABAD',
       'FATEHABAD', 'GURGAON', 'HISAR', 'JHAJJAR', 'JIND', 'KAITHAL',
       'KARNAL', 'KURUKSHETRA', 'MAHENDRAGARH', 'PANCHKULA', 'PANIPAT',
       'REWARI', 'ROHTAK', 'SIRSA', 'SONIPAT', 'YAMUNANAGAR', 'KANGRA',
       'KULLU', 'MANDI', 'SHIMLA', 'SOLAN', 'UNA', 'CHAMBA', 'HAMIRPUR',
       'SIRMAUR', 'KINNAUR', 'LAHUL AND SPITI', 'DODA', 'JAMMU', 'KATHUA',
       'RAJAURI', 'UDHAMPUR', 'KARGIL', 'LEH LADAKH', 'SRINAGAR',
       'BADGAM', 'BARAMULLA', 'POONCH', 'PULWAMA', 'ANANTNAG', 'KUPWARA',
       'CHATRA', 'DUMKA', 'GARHWA', 'GODDA', 'GUMLA', 'HAZARIBAGH',
       'KODERMA', 'LATEHAR', 'LOHARDAGA', 'PAKUR', 'PALAMU', 'RANCHI',
       'SAHEBGANJ', 'SARAIKELA KHARSAWAN', 'SIMDEGA', 'WEST SINGHBHUM',
       'BOKARO', 'DEOGHAR', 'DHANBAD', 'EAST SINGHBUM', 'GIRIDIH',
       'JAMTARA', 'BAGALKOT', 'BANGALORE RURAL', 'BELGAUM', 'BELLARY',
       'BENGALURU URBAN', 'CHAMARAJANAGAR', 'CHIKMAGALUR', 'CHITRADURGA',
       'DAKSHIN KANNAD', 'DAVANGERE', 'DHARWAD', 'GADAG', 'HASSAN',
       'HAVERI', 'KODAGU', 'KOLAR', 'MANDYA', 'MYSORE', 'SHIMOGA',
       'TUMKUR', 'UDUPI', 'UTTAR KANNAD', 'BIDAR', 'BIJAPUR', 'GULBARGA',
       'KOPPAL', 'RAICHUR', 'ALAPPUZHA', 'ERNAKULAM', 'IDUKKI', 'KANNUR',
       'KASARAGOD', 'KOLLAM', 'KOTTAYAM', 'KOZHIKODE', 'MALAPPURAM',
       'PALAKKAD', 'PATHANAMTHITTA', 'THIRUVANANTHAPURAM', 'THRISSUR',
       'WAYANAD', 'ANUPPUR', 'ASHOKNAGAR', 'BALAGHAT', 'BARWANI', 'BETUL',
       'BHIND', 'BHOPAL', 'BURHANPUR', 'CHHATARPUR', 'CHHINDWARA',
       'DAMOH', 'DATIA', 'DEWAS', 'DHAR', 'DINDORI', 'GUNA', 'GWALIOR',
       'HARDA', 'HOSHANGABAD', 'INDORE', 'JABALPUR', 'JHABUA', 'KATNI',
       'KHANDWA', 'KHARGONE', 'MANDLA', 'MANDSAUR', 'MORENA',
       'NARSINGHPUR', 'NEEMUCH', 'PANNA', 'RAISEN', 'RAJGARH', 'RATLAM',
       'REWA', 'SAGAR', 'SATNA', 'SEHORE', 'SEONI', 'SHAHDOL', 'SHAJAPUR',
       'SHEOPUR', 'SHIVPURI', 'SIDHI', 'TIKAMGARH', 'UJJAIN', 'UMARIA',
       'VIDISHA', 'AHMEDNAGAR', 'AKOLA', 'AMRAVATI', 'BEED', 'BHANDARA',
       'BULDHANA', 'CHANDRAPUR', 'DHULE', 'GADCHIROLI', 'GONDIA',
       'HINGOLI', 'JALGAON', 'JALNA', 'KOLHAPUR', 'LATUR', 'NAGPUR',
       'NANDED', 'NANDURBAR', 'NASHIK', 'OSMANABAD', 'PARBHANI', 'PUNE',
       'RAIGAD', 'RATNAGIRI', 'SANGLI', 'SATARA', 'SOLAPUR', 'THANE',
       'WARDHA', 'WASHIM', 'YAVATMAL', 'SINDHUDURG', 'SENAPATI',
       'BISHNUPUR', 'CHANDEL', 'CHURACHANDPUR', 'IMPHAL EAST',
       'IMPHAL WEST', 'TAMENGLONG', 'THOUBAL', 'UKHRUL',
       'EAST GARO HILLS', 'EAST JAINTIA HILLS', 'EAST KHASI HILLS',
       'RI BHOI', 'SOUTH GARO HILLS', 'WEST GARO HILLS',
       'WEST KHASI HILLS', 'AIZAWL', 'CHAMPHAI', 'KOLASIB', 'LUNGLEI',
       'MAMIT', 'SAIHA', 'DIMAPUR', 'KOHIMA', 'MOKOKCHUNG', 'MON', 'PHEK',
       'TUENSANG', 'WOKHA', 'ZUNHEBOTO', 'ANUGUL', 'BALANGIR',
       'BALESHWAR', 'BARGARH', 'BHADRAK', 'BOUDH', 'CUTTACK', 'DEOGARH',
       'DHENKANAL', 'GAJAPATI', 'GANJAM', 'JAGATSINGHAPUR', 'JAJAPUR',
       'JHARSUGUDA', 'KALAHANDI', 'KANDHAMAL', 'KENDRAPARA', 'KENDUJHAR',
       'KHORDHA', 'KORAPUT', 'MALKANGIRI', 'MAYURBHANJ', 'NABARANGPUR',
       'NAYAGARH', 'NUAPADA', 'RAYAGADA', 'SAMBALPUR', 'SONEPUR',
       'SUNDARGARH', 'PURI', 'MAHE', 'PONDICHERRY', 'KARAIKAL', 'YANAM',
       'AMRITSAR', 'BATHINDA', 'FARIDKOT', 'FATEHGARH SAHIB', 'FIROZEPUR',
       'HOSHIARPUR', 'JALANDHAR', 'KAPURTHALA', 'LUDHIANA', 'MOGA',
       'MUKTSAR', 'NAWANSHAHR', 'PATIALA', 'RUPNAGAR', 'SANGRUR', 'MANSA',
       'GURDASPUR', 'AJMER', 'ALWAR', 'BANSWARA', 'BARAN', 'BHARATPUR',
       'BHILWARA', 'BIKANER', 'BUNDI', 'CHITTORGARH', 'DAUSA', 'DHOLPUR',
       'DUNGARPUR', 'GANGANAGAR', 'HANUMANGARH', 'JAIPUR', 'JAISALMER',
       'JALORE', 'JHALAWAR', 'KARAULI', 'KOTA', 'NAGAUR', 'PALI',
       'RAJSAMAND', 'SAWAI MADHOPUR', 'SIKAR', 'SIROHI', 'TONK',
       'UDAIPUR', 'BARMER', 'CHURU', 'JHUNJHUNU', 'JODHPUR', 'PRATAPGARH',
       'EAST DISTRICT', 'NORTH DISTRICT', 'SOUTH DISTRICT',
       'WEST DISTRICT', 'COIMBATORE', 'DHARMAPURI', 'DINDIGUL', 'ERODE',
       'KANNIYAKUMARI', 'KARUR', 'KRISHNAGIRI', 'NAGAPATTINAM',
       'NAMAKKAL', 'PERAMBALUR', 'SALEM', 'THANJAVUR', 'THE NILGIRIS',
       'THENI', 'THIRUVARUR', 'TIRUCHIRAPPALLI', 'TIRUNELVELI',
       'VIRUDHUNAGAR', 'CUDDALORE', 'KANCHIPURAM', 'MADURAI',
       'PUDUKKOTTAI', 'RAMANATHAPURAM', 'SIVAGANGA', 'THIRUVALLUR',
       'THOOTHUKUDI', 'TIRUVANNAMALAI', 'VELLORE', 'VILLUPURAM', 'DHALAI',
       'NORTH TRIPURA', 'SOUTH TRIPURA', 'WEST TRIPURA', 'AGRA',
       'ALIGARH', 'ALLAHABAD', 'AMBEDKAR NAGAR', 'AMROHA', 'AURAIYA',
       'AZAMGARH', 'BAGHPAT', 'BAHRAICH', 'BALLIA', 'BALRAMPUR', 'BANDA',
       'BARABANKI', 'BAREILLY', 'BASTI', 'BIJNOR', 'BUDAUN',
       'BULANDSHAHR', 'CHANDAULI', 'CHITRAKOOT', 'DEORIA', 'ETAH',
       'ETAWAH', 'FAIZABAD', 'FARRUKHABAD', 'FATEHPUR', 'FIROZABAD',
       'GAUTAM BUDDHA NAGAR', 'GHAZIABAD', 'GHAZIPUR', 'GONDA',
       'GORAKHPUR', 'HARDOI', 'HATHRAS', 'JALAUN', 'JAUNPUR', 'JHANSI',
       'KANNAUJ', 'KANPUR DEHAT', 'KANPUR NAGAR', 'KAUSHAMBI', 'KHERI',
       'KUSHI NAGAR', 'LALITPUR', 'LUCKNOW', 'MAHARAJGANJ', 'MAHOBA',
       'MAINPURI', 'MATHURA', 'MAU', 'MEERUT', 'MIRZAPUR', 'MORADABAD',
       'MUZAFFARNAGAR', 'PILIBHIT', 'RAE BARELI', 'RAMPUR', 'SAHARANPUR',
       'SANT KABEER NAGAR', 'SANT RAVIDAS NAGAR', 'SHAHJAHANPUR',
       'SHRAVASTI', 'SIDDHARTH NAGAR', 'SITAPUR', 'SONBHADRA',
       'SULTANPUR', 'UNNAO', 'VARANASI', 'BAHRAI'))
    fig2 = plt.figure(figsize=(10, 4))
    if inpu:
        filtered_df = data[(data['District'] == inpu) ]
        sns.barplot(x='District', y='Yield', data=filtered_df,hue=filtered_df['Crop'])
        st.pyplot(fig2)
    fig1 = plt.figure(figsize=(10, 4))
    st.markdown(f"<h4 style='color:black;'>Crop vs Yield</h4>", unsafe_allow_html=True)
    x=st.selectbox("Select Crop",('Arecanut', 'Banana', 'Black pepper', 'Cashewnut', 'Coconut',
       'Dry chillies', 'Ginger', 'Other Kharif pulses', 'other oilseeds',
       'Rice', 'Sugarcane', 'Sweet potato', 'Arhar/Tur', 'Bajra',
       'Castor seed', 'Coriander', 'Cotton(lint)', 'Gram', 'Groundnut',
       'Horse-gram', 'Jowar', 'Linseed', 'Maize', 'Mesta',
       'Moong(Green Gram)', 'Niger seed', 'Onion', 'Other Rabi pulses',
       'Potato', 'Ragi', 'Rapeseed &Mustard', 'Safflower', 'Sesamum',
       'Small millets', 'Soyabean', 'Sunflower', 'Tapioca', 'Tobacco',
       'Turmeric', 'Urad', 'Wheat', 'Oilseeds total', 'Jute', 'Masoor',
       'Peas & beans (Pulses)', 'Barley', 'Garlic', 'Khesari', 'Sannhamp',
       'Guar seed', 'Moth', 'Cardamom'))
    if x:
        filtered_df = data[(data['Crop'] == x)]
        sns.barplot(x='Crop', y='Yield', data=filtered_df,hue=filtered_df['Season'])
        st.pyplot(fig1)

    