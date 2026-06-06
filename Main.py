"""
Fluxo:
  1. Seleciona imagem original e referência via interface gráfica
  2. Converte ambas para escala de cinza
  3. Aplica equalização de histograma
  4. Aplica especificação de histograma (matching com a referência)
  5. Aplica equalização com máscara na região agrícola
  6. Gera relatório comparativo com histogramas antes/depois
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import exposure, io, color, img_as_float
from tkinter import Tk, filedialog



# Seleçao de arquivos
def selecionar_imagem(titulo: str) -> str:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    caminho = filedialog.askopenfilename(
        title=titulo,
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")]
    )
    root.destroy()
    if not caminho:
        raise SystemExit("Nenhuma imagem selecionada. Encerrando.")
    return caminho

# Carregamento e pré processamento
def carregar_cinza(caminho: str) -> np.ndarray:
    #Carrega a imagem e converte em nivel de cinza
    img = io.imread(caminho)
    if img.ndim == 3:
        img = color.rgb2gray(img[:, :, :3])  # ignora canal alpha se existir
    return img_as_float(img)

# Processamentos
def equalizar_histograma(img: np.ndarray) -> np.ndarray:
    return exposure.equalize_hist(img)

def especificar_histograma(img: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    return exposure.match_histograms(img, referencia)

def criar_mascara_agricola(shape: tuple) -> np.ndarray:
    #Mascara, cobre o terço superior esquerdo, mas pode ser ajustado com as coordenadas da imagem real
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    mask[0:h//2, 0:w//2] = True
    return mask

def equalizar_com_mascara(img: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    #Equaliza apenas os pixels dentro da máscara
    img_out = img.copy()
    pixels = img[mascara]
    hist, bins = np.histogram(pixels, bins=256, range=(0, 1))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    lut = np.clip((cdf - cdf_min) / (mascara.sum() - cdf_min), 0, 1)
    indices = np.clip(np.searchsorted(bins[:-1], pixels, side="right") - 1, 0, 255)
    img_out[mascara] = lut[indices]
    return img_out

# Visualizaçao
def plot_hist(ax, img: np.ndarray, titulo: str, cor: str):
    hist, bins = exposure.histogram(img)
    ax.fill_between(bins, hist, alpha=0.5, color=cor)
    ax.plot(bins, hist, color=cor, linewidth=1.2)
    ax.set_title(titulo, fontsize=9, fontweight="bold")
    ax.set_xlabel("Intensidade")
    ax.set_ylabel("Frequência")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

def gerar_relatorio(img_orig, img_ref, img_eq, img_esp, img_eq_mask, mascara):
    cmap = "gray"
    fig = plt.figure(figsize=(20, 20))
    fig.patch.set_facecolor("#f5f5f5")
    fig.suptitle(
        "Relatório — Melhoria de Imagens de Satélite | PDI",
        fontsize=15, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.6, wspace=0.35,
                           left=0.05, right=0.97, top=0.95, bottom=0.03)

    def add_img(row, col, img, titulo):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(titulo, fontsize=9, fontweight="bold")
        ax.axis("off")
        return ax

    def add_hist(row, col, img, titulo, cor):
        ax = fig.add_subplot(gs[row, col])
        plot_hist(ax, img, titulo, cor)
        return ax

    # Linha 0: Original
    add_img(0, 0, img_orig, "Original (cinza)")
    add_hist(0, 1, img_orig, "Histograma — Original", "steelblue")

    # Linha 1: Referência
    add_img(1, 0, img_ref, "Referência (cinza)")
    add_hist(1, 1, img_ref, "Histograma — Referência", "darkorange")

    # Linha 2: Equalização
    add_img(2, 0, img_eq, "Equalizada")
    add_hist(2, 1, img_orig, "Histograma — Antes da Equalização", "steelblue")
    add_hist(2, 2, img_eq,   "Histograma — Após Equalização",     "seagreen")

    ax_diff1 = fig.add_subplot(gs[2, 3])
    diff1 = np.abs(img_eq - img_orig)
    im1 = ax_diff1.imshow(diff1, cmap="hot", vmin=0, vmax=0.5)
    ax_diff1.set_title("Diferença |Eq − Original|", fontsize=9, fontweight="bold")
    ax_diff1.axis("off")
    plt.colorbar(im1, ax=ax_diff1, fraction=0.046)

    # Linha 3: Especificação
    add_img(3, 0, img_esp, "Especificada (matching c/ ref.)")
    add_hist(3, 1, img_orig, "Histograma — Antes da Especificação", "steelblue")
    add_hist(3, 2, img_esp,  "Histograma — Após Especificação",     "purple")

    ax_ov = fig.add_subplot(gs[3, 3])
    for im_d, label, cor in [
        (img_orig, "Original",    "steelblue"),
        (img_ref,  "Referência",  "darkorange"),
        (img_esp,  "Especificada","purple"),
    ]:
        h, b = exposure.histogram(im_d)
        ax_ov.plot(b, h / h.max(), label=label, linewidth=1.4, alpha=0.85, color=cor)
    ax_ov.set_title("Sobreposição dos Histogramas", fontsize=9, fontweight="bold")
    ax_ov.set_xlabel("Intensidade")
    ax_ov.legend(fontsize=8)
    ax_ov.grid(alpha=0.3)

    # Linha 4: Máscara
    ax_mask = fig.add_subplot(gs[4, 0])
    vis = np.zeros((*mascara.shape, 3))
    vis[mascara]  = [0.2, 0.8, 0.2]
    vis[~mascara] = [0.85, 0.85, 0.85]
    ax_mask.imshow(vis)
    ax_mask.set_title("Máscara — Região Agrícola", fontsize=9, fontweight="bold")
    ax_mask.axis("off")

    add_img(4, 1, img_eq_mask, "Equalização c/ Máscara")
    add_hist(4, 2, img_orig[mascara],    "Histograma — Antes (região mascarada)", "steelblue")
    add_hist(4, 3, img_eq_mask[mascara], "Histograma — Após (região mascarada)",  "olivedrab")

    plt.savefig("relatorio_satelite.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("✅  Relatório salvo em: relatorio_satelite.png")
    plt.show()

# Pipeline princiapal
def main():
    print("Selecione a imagem ORIGINAL (baixo contraste)...")
    caminho_orig = selecionar_imagem("Selecione a imagem original")

    print("Selecione a imagem de REFERÊNCIA (bem iluminada)...")
    caminho_ref = selecionar_imagem("Selecione a imagem de referência")

    print("Carregando e convertendo para escala de cinza...")
    img_orig = carregar_cinza(caminho_orig)
    img_ref  = carregar_cinza(caminho_ref)

    print("Aplicando equalização de histograma...")
    img_eq = equalizar_histograma(img_orig)

    print("Aplicando especificação de histograma...")
    img_esp = especificar_histograma(img_orig, img_ref)

    print("Aplicando equalização com máscara agrícola...")
    mascara     = criar_mascara_agricola(img_orig.shape)
    img_eq_mask = equalizar_com_mascara(img_orig, mascara)

    print("Gerando relatório...")
    gerar_relatorio(img_orig, img_ref, img_eq, img_esp, img_eq_mask, mascara)


if __name__ == "__main__":
    main()
