from normalizacao import GaussNorm


def rodar_rotina():
    means = [[0, 1, 1, 5, 2, 8, 4, 5, 6, 1]]
    stds = [[11, 10, 3, 5, 2, 8, 4, 12, 4, 1]]
    noise_ranges = [[-1, 1], [-10, 3], [-1, 5], [-3, 7], [10, 30]]

    for norm in range(1, 5):
        for i in range(7):
            print norm, i
            nome_arquivo = '/home/pedroceles/Dropbox/Mestrado/Projeto/inicio/imagens/multiplas_dimensoes/norma_{}_back_{}'.format(norm, i + 1)
            titulo = "Norma: {}, Backbones: {}".format(norm, i + 1)
            GaussNorm.run(20, means=means, stds=stds, using_axis=range(i + 1), noise_ranges=noise_ranges, n_neighbors=3, size=10000, std_norm=norm, title=titulo, salvar_figura=nome_arquivo, plot=False)
