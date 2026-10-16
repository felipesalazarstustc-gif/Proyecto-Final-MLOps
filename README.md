# Proyecto Final MLOps

**Estudiante:** Felipe  
**Variante:** A - Regresión

## Descripción

Este proyecto consiste en entrenar y monitorear un modelo de Machine Learning para predecir la cantidad de bicicletas alquiladas usando datos históricos.

Se utilizan dos modelos: Ridge Regression y Gradient Boosting.

## Herramientas utilizadas

- Python
- Flask
- MLflow
- Docker
- Hyperopt
- Scikit-learn

## Funcionamiento

El proyecto permite preparar los datos, entrenar modelos, comparar resultados y guardar los modelos con MLflow.

También cuenta con un sistema de monitoreo para detectar cambios en los datos y un proceso de reentrenamiento.

El modelo se puede utilizar mediante una API creada con Flask.

## Cómo ejecutar

1. Tener Docker instalado.
2. Copiar `.env.example` y renombrar la copia a `.env`.
3. Abrir una terminal en la carpeta del proyecto.
4. Ejecutar:

```bash
docker compose up -d --build
```

## Carpetas principales

- `data`: preparación de datos.
- `training`: entrenamiento de modelos.
- `registry`: registro y promoción de modelos.
- `pipeline`: proceso de entrenamiento.
- `serving`: API para predicciones.
- `monitoring`: monitoreo de datos.
- `tests`: pruebas del proyecto.

Proyecto realizado para la asignatura de MLOps, utilizando la variante A de regresión.
