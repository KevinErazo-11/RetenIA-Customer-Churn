# Instalación limpia de FaroCX V2 en Ubuntu

La versión nueva **no** reutiliza modelos ni métricas anteriores. **Se debe entrenar con el CSV oficial**.

1. Detén Streamlit con `Ctrl+C` y cierra el notebook anterior.
2. Conserva el trabajo previo renombrando `V2` a `V2_respaldo` (no lo borres).
3. Extrae el ZIP en el directorio que contiene `V2_respaldo`. El ZIP crea una única carpeta `V2`.
4. Opcional: si conservas el CSV auténtico en el respaldo, cópialo desde `V2_respaldo/data/WA_Fn-UseC_-Telco-Customer-Churn.csv` hasta `V2/data/`. **No** copies `artifacts/` ni `figures/` desde el respaldo, porque corresponden a un entrenamiento distinto.
5. Entra en `V2` y activa el ambiente virtual existente:

```bash
source "/home/kevin-erazo/Documentos/Ambiente Virtual/bin/activate"
PROJECT="/home/kevin-erazo/OneDrive/8) Octavo Semestre/Aprendizaje Automatico/7. Optimización de Machine Learning, Explainable AI y Business Pitch/V2"
cd "$PROJECT"
python -m pip install -r requirements.txt
python -m pip check
python run_experiments.py
python build_pitch.py
pdfinfo Pitch_FaroCX_Kevin_Erazo.pdf | grep Pages
python VALIDAR_ENTREGA.py
python -m streamlit run app.py
```

El PDF debe mostrar `Pages: 5`. La app se abre en `http://localhost:8501`.

El notebook está en `notebooks/02_Tarea7_Proyecto_Completo_Kevin_Erazo.ipynb`. Abre la carpeta `V2` completa en VS Code y usa el kernel del ambiente virtual. Selecciona **Restart / Run All** para reproducir el entrenamiento en el notebook y guarda las salidas. Esa acción vuelve a ejecutar los experimentos; para poner Streamlit en marcha **no es necesario** repetirla después del script.

El paquete no trae pesos ya entrenados ni resultados de una nueva ejecución. El conjunto de prueba se usa al final de cada entrenamiento. Si hay un fallo, guarda el traceback completo; no ejecutes `--quick` como resultado evaluable.

El despliegue GitHub/Streamlit Cloud y la URL pública continúan siendo acciones externas que requieren tu cuenta. No entregues `localhost` como enlace al profesor.
