# 🏦 MVP: Evaluación Automatizada de Crédito y Marco Legal de IA en Colombia

Este repositorio contiene un Producto Mínimo Viable (MVP) desarrollado para la asignatura de **Programación Avanzada**. Consiste en un modelo de red neuronal profunda implementado en **PyTorch** con una interfaz gráfica en **Gradio**, orientado a la evaluación automatizada de solicitudes de crédito y alineado con la normativa colombiana de protección de datos personales.

---

## 📋 Descripción del Proyecto

El sistema evalúa variables financieras y demográficas de un usuario para predecir la probabilidad de aprobación de un crédito mediante clasificación binaria. Se enfoca no solo en la precisión técnica, sino también en el cumplimiento del derecho a la explicabilidad y la protección de datos sensibles en decisiones automatizadas.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.13+
* **Framework DL:** PyTorch (`torch`)
* **Preprocesamiento:** Scikit-Learn (`StandardScaler`)
* **Procesamiento de Datos:** NumPy
* **Interfaz de Usuario:** Gradio

---

## ⚖️ Marco Legal Colombiano Aplicado

El desarrollo contempla las siguientes normativas:
* **Ley 1581 de 2012 (Habeas Data):** Garantía del derecho del titular a conocer, actualizar y rectificar la información recolectada en bases de datos.
* **Circular 002 de 2024 (SIC):** Transparencia, no discriminación y supervisión humana en la toma de decisiones basada en algoritmos de Inteligencia Artificial.

---

## 🚀 Instalación y Ejecución

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/mvp_redes_neuronales.git](https://github.com/Nandoreyes/mvp_redes_neuronales.git)
   cd mvp_redes_neuronales

  
  
  
2. Crear y activar el entorno virtual:

PowerShell
python -m venv venv

# En Windows (PowerShell):
.\venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate






3. Instalar dependencias:

Bash
pip install torch numpy scikit-learn gradio

4. Ejecutar la aplicación:

Bash
python app.py




💻 Arquitectura de la Red Neuronal
Capa de Entrada: 4 características (Edad, Ingresos, Historial Crediticio, Deuda).

Capa Oculta 1: 16 neuronas con activación ReLU.

Capa Oculta 2: 8 neuronas con activación ReLU.

Capa de Salida: 1 neurona con activación Sigmoid (salida probabilística entre 0 y 1).

Optimización: Optimizador Adam con función de pérdida BCELoss (Binary Cross-Entropy).
