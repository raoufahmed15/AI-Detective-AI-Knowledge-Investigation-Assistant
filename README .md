# 🕵️ AI Detective — Streamlit Deployment

واجهة Streamlit لمشروع **AI Detective** (RAG على تقارير PDF + سجلات الحوادث المنظمة).
الجزء الخاص بالـ Retrieval (embeddings + FAISS) هو **نفسه بالظبط** اللي في النوت بوك.
الجزء الخاص بتوليد الإجابة بس اتغيّر من موديل محلي (Mistral-Nemo, 12B) إلى **Google
Gemini API**، عشان الموديل المحلي مش هيشتغل على استضافة زي Streamlit Cloud (مفيهاش GPU).

## 1. هيكل الملفات المطلوب

حط مجلد الآرتيفاكتس (اللي شكله زي الصورة اللي بعتها: `config.json` / `index.faiss` /
`metadata.pkl`) جنب `app.py` بنفس الاسم `model`:

```
your-repo/
├── app.py
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml        # (متعملش commit له — سر)
└── model/
    ├── config.json
    ├── index.faiss
    └── metadata.pkl
```

لو المجلد عندك اسمه حاجة تانية غير `model`، غيّر قيمة `ARTIFACTS_DIR` في أول ملف
`app.py` (سطر واحد) — مفيش أي إعداد ظاهر لليوزر في الواجهة، كل حاجة متحكم فيها من الكود.

## 2. احصل على مفتاح Google Gemini API

من https://aistudio.google.com/apikey (مجاني، بحد يومي محدود) — هتحتاجه في الخطوة الجاية.

## 3. ارفع المشروع على GitHub

```bash
git init
git add .
git commit -m "AI Detective Streamlit app"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

تأكد إن مجلد `model/` (بالـ 3 ملفات) اتعمله push فعليًا (ملفات الـ FAISS والـ pickle
مش كبيرة عادة فمش هتواجه مشكلة مع حدود GitHub).

## 4. النشر على Streamlit Community Cloud

1. https://share.streamlit.io → **New app**
2. اختَر الـ repo والفرع، وحدد `app.py` كملف رئيسي
3. من **Advanced settings → Secrets** حط:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
   ومن نفس الـ Advanced settings اختار **Python 3.11** من قائمة الـ Python version
   (مهم علشان faiss-cpu يتثبت صح).
4. Deploy — وخلاص، هيبقى عندك لينك لايف.

## 5. تشغيل محلي (اختياري، لو عايز تجرب قبل الرفع)

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # وحط مفتاحك فيه
streamlit run app.py
```

## ملاحظات مهمة

- **أول تحميل** للموديل `intfloat/multilingual-e5-base` بياخد شوية وقت (بينزّل الأوزان
  مرة واحدة) وبعدين بيتخزن في الكاش (`st.cache_resource`).
- استضافة Streamlit Cloud المجانية بتوفر ~1GB RAM؛ لو حسّيت إن التطبيق بطيء أو بيقفل،
  فكّر في خطة مدفوعة أو استضافة تانية (مثل Hugging Face Spaces بـ CPU أكبر).
- الأسئلة والإجابات كلها بتتبني حصريًا على الأدلة (evidence) المسترجعة من الـ FAISS —
  الموديل متأمَر إنه ميخترعش حقائق، وأي تفصيلة مشتركة بين حالتين (زي نفس العربية) بيتقال
  عليها "احتمال ربط" مش "دليل قاطع"، بالظبط زي في النوت بوك.