from config import *
import matplotlib.pyplot as plt
import numpy as np
import uuid


def plot_training_history(history, hype_space, model_name=None):
    """Plota gráficos de acurácia e loss do treinamento."""
    plt.figure(figsize=(12, 5))
    
    # Gráfico de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia por Época')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend()
    
    # Gráfico de Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss por Época')
    plt.ylabel('Loss')
    plt.xlabel('Época')
    plt.legend()
    
    # Adicionar informações do trial ao gráfico
    params_str = "\n".join([f"{k}: {v}" for k, v in hype_space.items()])
    plt.suptitle(f"Parâmetros do Trial:\n{params_str}", y=1.05)
    
    # Ajustar layout e salvar
    plt.tight_layout()
    
    # Criar nome do arquivo baseado no model_name ou gerar um UUID
    if model_name:
        # Extrai apenas a parte do ID do modelo (remove o prefixo "model_")
        model_id = model_name.split('_')[-1]
        plot_filename = f"training_plot_{model_id}.png"
    else:
        plot_filename = f"training_plot_{str(uuid.uuid4())[:8]}.png"
    
    plot_path = os.path.join(RESULTS_DIR, plot_filename)
    
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Gráfico de treinamento salvo em: {plot_path}")
    return plot_path  # Retorna o caminho para possível uso posterior
