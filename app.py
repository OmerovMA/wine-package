from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
from wine_quality_model.predict import make_prediction

app = FastAPI()

# Подключаем папки с шаблонами и статикой
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница с формой"""
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict/", response_class=HTMLResponse)
async def predict_wine_quality(
    request: Request,
    fixed_acidity: float = Form(
        default=7.4,
        gt=4.0,
        lt=16.0,
        description="Fixed acidity (4.0-16.0 g/L)"
    ),
    volatile_acidity: float = Form(
        default=0.7,
        gt=0.1,
        lt=1.5,
        description="Volatile acidity (0.1-1.5 g/L)"
    ),
    citric_acid: float = Form(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Citric acid (0.0-1.0 g/L)"
    ),
    residual_sugar: float = Form(
        default=1.9,
        ge=0.5,
        le=30.0,
        description="Residual sugar (0.5-30.0 g/L)"
    ),
    chlorides: float = Form(
        default=0.076,
        ge=0.01,
        le=0.2,
        description="Chlorides (0.01-0.2 g/L)"
    ),
    free_sulfur_dioxide: float = Form(
        default=11.0,
        ge=1,
        le=100,
        description="Free sulfur dioxide (1-100 mg/L)"
    ),
    total_sulfur_dioxide: float = Form(
        default=34.0,
        ge=5,
        le=200,
        description="Total sulfur dioxide (5-200 mg/L)"
    ),
    density: float = Form(
        default=0.9978,
        gt=0.98,
        lt=1.04,
        description="Density (0.98-1.04 g/cm³)"
    ),
    ph: float = Form(
        default=3.51,
        gt=2.5,
        lt=4.0,
        description="pH (2.5-4.0)"
    ),
    sulphates: float = Form(
        default=0.56,
        gt=0.3,
        lt=2.0,
        description="Sulphates (0.3-2.0 g/L)"
    ),
    alcohol: float = Form(
        default=9.4,
        gt=8.0,
        lt=15.0,
        description="Alcohol (8.0-15.0 % vol)"
    )
):
    """Обработка формы и вывод результата"""
    input_data = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    # Делаем предсказание
    prediction = make_prediction(input_data=pd.DataFrame(input_data, index=[0]))

    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "prediction": prediction["predictions"][0],
            "input_data": input_data  # Чтобы сохранить введенные значения
        }
    )