import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gradio as gr

# ---------------------------------------------------------
# 1. PREPROCESAMIENTO Y GENERACIÓN DE DATOS
# ---------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)
num_samples = 1000

# Variables: [Edad, Ingresos Mensuales (COP), Score Historial (0 a 1), Deuda Total (COP)]
edad = np.random.randint(18, 70, size=(num_samples, 1))
ingresos = np.random.randint(1500000, 15000000, size=(num_samples, 1))
historial = np.random.rand(num_samples, 1)
deuda = np.random.randint(0, 10000000, size=(num_samples, 1))

X = np.hstack((edad, ingresos, historial, deuda))

# Criterio objetivo simulado
score = (ingresos * 0.4 + historial * 5000000 - deuda * 0.5)
y = (score > 2000000).astype(np.float32).reshape(-1, 1)

# Normalización / Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convertir datos a Tensores de PyTorch
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val)

# ---------------------------------------------------------
# 2. ARQUITECTURA DE LA RED NEURONAL (PyTorch)
# ---------------------------------------------------------
class RedNeuronalCredito(nn.Module):
    def __init__(self):
        super(RedNeuronalCredito, self).__init__()
        self.capa1 = nn.Linear(4, 16)      # Capa Entrada -> Oculta 1
        self.relu1 = nn.ReLU()
        self.capa2 = nn.Linear(16, 8)      # Capa Oculta 1 -> Oculta 2
        self.relu2 = nn.ReLU()
        self.capa_salida = nn.Linear(8, 1) # Capa Oculta 2 -> Salida
        self.sigmoid = nn.Sigmoid()        # Función de activación para probabilidad (0 a 1)

    def forward(self, x):
        x = self.relu1(self.capa1(x))
        x = self.relu2(self.capa2(x))
        x = self.sigmoid(self.capa_salida(x))
        return x

model = RedNeuronalCredito()
criterion = nn.BCELoss()  # Binary Cross-Entropy (Función de costo/error)
optimizer = optim.Adam(model.parameters(), lr=0.01) # Optimizador Adam

# Ciclo de entrenamiento
print("Entrenando la red neuronal en PyTorch...")
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()  # Retropropagación (Backpropagation)
    optimizer.step() # Actualización de pesos
print("¡Entrenamiento completado exitosamente!")

# ---------------------------------------------------------
# 3. LÓGICA DE INFERENCIA
# ---------------------------------------------------------
def evaluar_credito(edad_user, ingresos_user, historial_user, deuda_user):
    datos = np.array([[edad_user, ingresos_user, historial_user, deuda_user]], dtype=np.float32)
    datos_scaled = scaler.transform(datos)
    datos_tensor = torch.tensor(datos_scaled)
    
    model.eval()
    with torch.no_grad():
        probabilidad = float(model(datos_tensor)[0][0])
    
    aprobado = probabilidad >= 0.5
    dict_resultado = {
        "Aprobado": probabilidad,
        "Rechazado": 1.0 - probabilidad
    }
    
    estado = " CRÉDITO APROBADO" if aprobado else " CRÉDITO RECHAZADO"
    info_legal = "\n\n*Cumplimiento Ley 1581/2012:* Decisión generada mediante procesamiento automatizado. El usuario conserva derecho a revisión y rectificación."
    
    return f"{estado} (Probabilidad: {probabilidad*100:.1f}%){info_legal}", dict_resultado

# ---------------------------------------------------------
# 4. INTERFAZ GRÁFICA INTERACTIVA (GRADIO)
# ---------------------------------------------------------
interface = gr.Interface(
    fn=evaluar_credito,
    inputs=[
        gr.Slider(18, 80, value=30, label="Edad"),
        gr.Number(value=3500000, label="Ingresos Mensuales (COP)"),
        gr.Slider(0.0, 1.0, value=0.8, label="Score de Historial Crediticio (0 a 1)"),
        gr.Number(value=1000000, label="Deuda Actual Total (COP)")
    ],
    outputs=[
        gr.Textbox(label="Resultado y Nota Legal"),
        gr.Label(label="Distribución de Probabilidad")
    ],
    title="MVP: Evaluación Automatizada de Crédito (PyTorch)",
    description="Red neuronal para clasificación binaria respetando el marco de protección de datos personales en Colombia."
)

if __name__ == "__main__":
    interface.launch(inbrowser=True)