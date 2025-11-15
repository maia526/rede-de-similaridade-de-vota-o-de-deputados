import requests
import json
import networkx as nx
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import csv
import os
import matplotlib.pyplot as plt
import scipy
import collections
try:
    import pandas as pd
except ImportError:
    pd = None

G = nx.Graph()
qtdVotacoes = 0

votacoes_por_deputado = {}
index_votacoes = {}
deputados = {}

def acharVotacoes():
    global qtdVotacoes
    pasta_votacoes = 'dadosProntos/votacoesDeputados'
    resultados = {}
    
    # Verifica se a pasta existe
    if not os.path.exists(pasta_votacoes):
        print(f"Pasta {pasta_votacoes} não encontrada.")
        return resultados
    
    # Lista todos os arquivos CSV na pasta
    arquivos_csv = [f for f in os.listdir(pasta_votacoes) if f.endswith('.csv')]
    for arquivo in arquivos_csv:
        caminho_arquivo = os.path.join(pasta_votacoes, arquivo)   
        
        with open(caminho_arquivo, 'r', encoding='utf-8') as csvfile:
            # O CSV usa ponto e vírgula como delimitador
            reader = csv.DictReader(csvfile, delimiter=';')
            
            for row in reader:
                idVotacao = str(row.get('\ufeff"idVotacao"'))
                if idVotacao in index_votacoes:
                    idDeputado = str(row.get('deputado_id'))
                    if idDeputado not in deputados:
                        deputados[idDeputado] = {
                            'nome': str(row.get('deputado_nome')),
                            'partido' : str(row.get('deputado_siglaPartido')),
                            'uf' : str(row.get('deputado_siglaUf'))
                        }
                        votacoes_por_deputado[idDeputado] = np.zeros(qtdVotacoes)
                    arrayDeputado = votacoes_por_deputado[idDeputado]
                    voto = str(row.get('voto'))
                    if voto == "Sim":
                        voto = 1
                    elif voto =="Não":
                        voto = -1
                    else: voto = 0
                    index = index_votacoes[idVotacao]
                    arrayDeputado[index] = voto
                    votacoes_por_deputado[idDeputado] = arrayDeputado


def contar_linhas_por_periodo():
    global qtdVotacoes
    pasta_votacoes = 'dadosProntos/votacoes'
    data_inicio = datetime(2019, 2, 1)
    data_fim = datetime(2023, 1, 31, 23, 59, 59)  # Inclui todo o dia 31/01/2023
    
    # Verifica se a pasta existe
    if not os.path.exists(pasta_votacoes):
        print(f"Pasta {pasta_votacoes} não encontrada.")
        return
    
    # Lista todos os arquivos CSV na pasta
    arquivos_csv = [f for f in os.listdir(pasta_votacoes) if f.endswith('.csv')]
    for arquivo in arquivos_csv:
        caminho_arquivo = os.path.join(pasta_votacoes, arquivo)
        contador = 0
    
        with open(caminho_arquivo, 'r', encoding='utf-8') as csvfile:
            # O CSV usa ponto e vírgula como delimitador
            reader = csv.DictReader(csvfile, delimiter=';')
            
            for row in reader:
                data_hora_str = row.get('dataHoraRegistro', '').strip()
                
                # Verifica se o campo não está vazio
                if data_hora_str:
                    try:
                        # Converte a string para datetime
                        # Formato: "YYYY-MM-DDTHH:MM:SS"
                        data_hora = datetime.strptime(data_hora_str, '%Y-%m-%dT%H:%M:%S')
                        
                        # Verifica se está no período especificado
                        if data_inicio <= data_hora <= data_fim:
                            qtdVotacoes+=1
                            idVotacao = str(row.get('\ufeff"id"'))
                            index_votacoes[idVotacao] = contador
                            contador += 1
                    except ValueError:
                        # Se não conseguir fazer o parse, ignora a linha
                        continue

def criarVertices():
    global deputados
    for deputado in deputados:
        atributos = deputados[str(deputado)]
        G.add_nodes_from([(deputado, atributos)])

def calcularArestas():
    """
    Calcula arestas entre vértices baseado na similaridade normalizada dos arrays de votações.
    A similaridade é calculada dividindo o produto interno (com -1/0/1) pelo produto interno
    com valores apenas positivos (0/1), resultando em uma medida normalizada entre -1 e +1.
    """
    vertices = list(G.nodes())
    
    for i, vertice1 in enumerate(vertices):
        # Pega o ID do vértice (o próprio nó é o ID)
        id_deputado1 = str(vertice1)
        
        # Pega o array de votações do primeiro deputado
        if id_deputado1 not in votacoes_por_deputado:
            continue
        
        array_votacoes1 = votacoes_por_deputado[id_deputado1]
        
        # Para cada outro vértice (excluindo ele mesmo)
        for j, vertice2 in enumerate(vertices):
            if i != j:  # Exclui ele mesmo
                id_deputado2 = str(vertice2)
                
                # Pega o array de votações do segundo deputado
                if id_deputado2 not in votacoes_por_deputado:
                    continue
                
                array_votacoes2 = votacoes_por_deputado[id_deputado2]
                
                # Calcula o produto interno normal (com -1/0/1) - mede similaridade considerando concordância/discordância
                produto_interno_similaridade = np.dot(array_votacoes1, array_votacoes2)
                
                # Cria versões normalizadas dos arrays (substitui -1 por 1, mantém 0 como 0)
                # Isso converte para apenas valores 0/1, onde 1 significa "votou" (independente de Sim/Não)
                array_votacoes1_normalizado = np.abs(array_votacoes1)  # -1 vira 1, 0 fica 0, 1 fica 1
                array_votacoes2_normalizado = np.abs(array_votacoes2)
                
                # Calcula o produto interno normalizado (apenas 0/1) - mede quantidade de participações conjuntas
                produto_interno_participacoes = np.dot(array_votacoes1_normalizado, array_votacoes2_normalizado)
                
                # Calcula a similaridade normalizada (divisão do primeiro pelo segundo)
                # Resultado entre -1 e +1: +1 = sempre concordam, 0 = misto, -1 = sempre discordam
                if produto_interno_participacoes > 0:
                    similaridade_normalizada = produto_interno_similaridade / produto_interno_participacoes
                else:
                    # Se não há participações conjuntas, similaridade é 0
                    similaridade_normalizada = 0.0
                
                if similaridade_normalizada >= 0.7:
                    # Adiciona aresta com a similaridade normalizada como peso
                    G.add_edge(vertice1, vertice2, weight=similaridade_normalizada)
    k = 0

def calcularArestasV2():
    vertices = list(G.nodes())
    
    for i, vertice1 in enumerate(vertices):
        # Pega o ID do vértice (o próprio nó é o ID)
        id_deputado1 = str(vertice1)
        
        # Pega o array de votações do primeiro deputado
        if id_deputado1 not in votacoes_por_deputado:
            continue
        
        array_votacoes1 = votacoes_por_deputado[id_deputado1]
        
        # Para cada outro vértice (excluindo ele mesmo)
        for j, vertice2 in enumerate(vertices):
            if i != j:  # Exclui ele mesmo
                id_deputado2 = str(vertice2)
                
                # Pega o array de votações do segundo deputado
                if id_deputado2 not in votacoes_por_deputado:
                    continue
                
                array_votacoes2 = votacoes_por_deputado[id_deputado2]
                if np.array_equal(array_votacoes1, array_votacoes2):
                    G.add_edge(vertice1, vertice2, weight=1)

def listar_partidos_unicos(grafo):
    """
    Extrai todos os partidos únicos dos nós do grafo e imprime a lista.
    
    Args:
        grafo: Grafo NetworkX com nós que possuem atributo 'partido'
    
    Returns:
        list: Lista de partidos únicos (sem duplicatas), ordenada alfabeticamente
    """
    partidos = set()
    
    for node in grafo.nodes():
        if 'partido' in grafo.nodes[node]:
            partido = grafo.nodes[node]['partido']
            if partido:  # Verifica se não é vazio/None
                partidos.add(partido)
    
    partidos_unicos = sorted(list(partidos))
    
    print("Lista de partidos únicos na rede:")
    print("-" * 40)
    for i, partido in enumerate(partidos_unicos, 1):
        print(f"{i}. {partido}")
    print("-" * 40)
    print(f"Total: {len(partidos_unicos)} partidos únicos")
    
    return partidos_unicos

def agrupar_partidos_por_classificacao(arquivo_ods='dadosProntos/posicionamentos.ods'):
    """
    Lê o arquivo posicionamentos.ods e agrupa os partidos por classificação.
    Cada classificação possível terá uma lista de partidos.
    
    Args:
        arquivo_ods: Caminho para o arquivo ODS (padrão: 'dadosProntos/posicionamentos.ods')
    
    Returns:
        dict: Dicionário onde as chaves são as classificações e os valores são listas de partidos
    """
    if pd is None:
        print("Erro: pandas não está instalado. Instale com: pip install pandas odfpy")
        return {}
    
    # Verifica se o arquivo existe
    if not os.path.exists(arquivo_ods):
        print(f"Erro: Arquivo {arquivo_ods} não encontrado.")
        return {}
    
    try:
        # Tenta ler o arquivo ODS usando pandas
        # Primeiro tenta com engine='odf' (requer odfpy)
        try:
            df = pd.read_excel(arquivo_ods, engine='odf')
        except:
            # Se falhar, tenta com ezodf
            try:
                df = pd.read_excel(arquivo_ods, engine='ezodf')
            except:
                # Se ainda falhar, tenta sem especificar engine
                df = pd.read_excel(arquivo_ods)
        
        # Identifica a coluna de classificação
        # Procura por colunas que possam conter classificação
        coluna_classificacao = None
        possiveis_nomes = ['classificação', 'classificacao', 'espectro', 'posicionamento', 
                          'categoria', 'tipo', 'classificacao', 'Classificação', 'Classificacao']
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(nome in col_lower for nome in possiveis_nomes):
                coluna_classificacao = col
                break
        
        # Se não encontrou, tenta a última coluna ou uma coluna específica
        if coluna_classificacao is None:
            # Tenta a última coluna
            if len(df.columns) > 1:
                coluna_classificacao = df.columns[-1]
            else:
                print("Erro: Não foi possível identificar a coluna de classificação.")
                print(f"Colunas disponíveis: {list(df.columns)}")
                return {}
        
        # Identifica a coluna de partido (geralmente a primeira ou uma com 'partido' no nome)
        coluna_partido = None
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'partido' in col_lower or 'sigla' in col_lower:
                coluna_partido = col
                break
        
        if coluna_partido is None:
            coluna_partido = df.columns[0]  # Usa a primeira coluna como padrão
        
        # Remove linhas com valores NaN
        df = df.dropna(subset=[coluna_classificacao, coluna_partido])
        
        # Agrupa os partidos por classificação
        partidos_por_classificacao = {}
        
        for _, row in df.iterrows():
            classificacao = str(row[coluna_classificacao]).strip()
            partido = str(row[coluna_partido]).strip()
            
            # Ignora valores vazios ou NaN
            if classificacao and partido and classificacao.lower() != 'nan' and partido.lower() != 'nan':
                if classificacao not in partidos_por_classificacao:
                    partidos_por_classificacao[classificacao] = []
                
                # Adiciona o partido se ainda não estiver na lista
                if partido not in partidos_por_classificacao[classificacao]:
                    partidos_por_classificacao[classificacao].append(partido)
        
        # Ordena as listas de partidos
        for classificacao in partidos_por_classificacao:
            partidos_por_classificacao[classificacao].sort()
        
        return partidos_por_classificacao
        
    except Exception as e:
        print(f"Erro ao ler o arquivo ODS: {e}")
        import traceback
        traceback.print_exc()
        return {}

def imprimir_partidos_por_classificacao(partidos_por_classificacao):
    """
    Imprime as listas de partidos agrupadas por classificação.
    
    Args:
        partidos_por_classificacao: Dicionário com classificações como chaves e listas de partidos como valores
    """
    if not partidos_por_classificacao:
        print("Nenhum dado para imprimir.")
        return
    
    print("=" * 80)
    print("PARTIDOS AGRUPADOS POR CLASSIFICAÇÃO")
    print("=" * 80)
    print()
    
    # Ordena as classificações alfabeticamente
    classificacoes_ordenadas = sorted(partidos_por_classificacao.keys())
    
    for classificacao in classificacoes_ordenadas:
        partidos = partidos_por_classificacao[classificacao]
        print(f"\n{'-' * 80}")
        print(f"{classificacao.upper()} ({len(partidos)} partidos)")
        print(f"{'-' * 80}")
        for i, partido in enumerate(partidos, 1):
            print(f"  {i}. {partido}")
    
    print("\n" + "=" * 80)
    total_partidos = sum(len(v) for v in partidos_por_classificacao.values())
    print(f"Total de classificações: {len(partidos_por_classificacao)}")
    print(f"Total de partidos: {total_partidos}")
    print("=" * 80)


def pintarVerticesPorOrientacaoPolitica():
    """
    Atribui cores aos vértices do grafo baseado na orientação política dos partidos.
    Retorna uma lista de cores correspondente à ordem dos nós do grafo.
    """
    extremaEsquerda = ['PSTU', 'PCO', 'PCB', 'PSOL']
    esquerda = ['PCdoB', 'PT']
    centroEsquerda = ['PDT', 'PSB']
    centro = ['Rede', 'PPS', 'PV', 'MDB', 'CIDADANIA', 'SOLIDARIEDADE']
    centroDireita = ['PTB', 'AVANTE', 'SDD', 'PMN', 'PMB', 'PHS', 'PP', 'UNIAO']
    direita = ['PMDB', 'PSD', 'PL', 'PSDB', 'PODEMOS', 'PODE', 'PPL', 'PRTB', 'PROS', 'PRB', 'PR', 'PTC', 'PSL', 'DC', 'PROGRESSISTAS', 'NOVO', 'PDC', 'PATRIOTA', 'PSC', 'REPUBLICANOS']
    extremaDireita = ['DEM']

    cores = {
        'Extrema-direita': 'blue', 
        'Direita': 'blue', 
        'Centro-direita': 'blue',   
        'Centro': 'yellow', 
        'Centro-esquerda': 'red', 
        'Esquerda': 'red', 
        'Extrema-esquerda': 'red',  
        'Sem classificação': 'gray'  
    }
    
    # Cria um dicionário para mapear partido -> cor
    partido_para_cor = {}
    
    # Mapeia cada partido para sua cor correspondente
    for partido in direita:
        partido_para_cor[partido] = cores['Direita']
    for partido in centroDireita:
        partido_para_cor[partido] = cores['Centro-direita']
    for partido in esquerda:
        partido_para_cor[partido] = cores['Esquerda']
    for partido in centroEsquerda:
        partido_para_cor[partido] = cores['Centro-esquerda']
    for partido in extremaDireita:
        partido_para_cor[partido] = cores['Extrema-direita']
    for partido in centro:
        partido_para_cor[partido] = cores['Centro']
    for partido in extremaEsquerda:
        partido_para_cor[partido] = cores['Extrema-esquerda']
    
    # Cria lista de cores para cada vértice do grafo
    cores_vertices = []
    for node in G.nodes():
        partido = G.nodes[node].get('partido', '')
        # Tenta encontrar a cor do partido, se não encontrar usa cinza
        cor = partido_para_cor.get(partido, cores['Sem classificação'])
        cores_vertices.append(cor)
    
    return cores_vertices

def obter_orientacao_politica(partido):
    """
    Retorna a orientação política de um partido como número:
    -1: extrema esquerda ou esquerda
    0: centro esquerda, centro direita e centro
    1: direita ou extrema direita
    """
    extremaEsquerda = ['PSTU', 'PCO', 'PCB', 'PSOL']
    esquerda = ['PCdoB', 'PT']
    centroEsquerda = ['PDT', 'PSB']
    centro = ['Rede', 'PPS', 'PV', 'MDB', 'CIDADANIA', 'SOLIDARIEDADE']
    centroDireita = ['PTB', 'AVANTE', 'SDD', 'PMN', 'PMB', 'PHS', 'PP', 'UNIAO']
    direita = ['PMDB', 'PSD', 'PL', 'PSDB', 'PODEMOS', 'PODE', 'PPL', 'PRTB', 'PROS', 'PRB', 'PR', 'PTC', 'PSL', 'DC', 'PROGRESSISTAS', 'NOVO', 'PDC', 'PATRIOTA', 'PSC', 'REPUBLICANOS']
    extremaDireita = ['DEM']
    
    if partido in extremaEsquerda:
        return -1
    elif partido in esquerda:
        return -0.6
    elif partido in centroEsquerda:
        return -0.3
    elif partido in centro:
        return 0
    elif partido in centroDireita:
        return 0.3
    elif partido in direita:
        return 0.6
    elif partido in extremaDireita:
        return 1
    else:
        return 0  # Default para centro se não encontrar

def clusterizacao_sem_heatmap_girvan_newman():
    """
    Gera o grafo de clusterização sem heatmap de fundo.
    """
    isolates = list(nx.isolates(G))
    G_temp = G.copy()
    G_temp.remove_nodes_from(isolates)
    comp = nx.algorithms.community.girvan_newman(G_temp)

    # Obter as primeiras divisões do grafo
    first_level = next(comp)      # Primeira divisão (2 comunidades)
    second_level = next(comp)     # Segunda divisão (mais fragmentada)

    communities = [list(c) for c in second_level]

    # Define as cores específicas para cada comunidade
    cores_comunidades = ['yellow', 'green', 'blue', 'purple', 'pink']  # Amarelo, Verde, Azul
    
    # Atribuir uma cor a cada comunidade
    colors = {}
    for i, comm in enumerate(communities):
        # Usa a cor correspondente ao índice, ou repete as cores se houver mais comunidades
        cor = cores_comunidades[i % len(cores_comunidades)]
        for node in comm:
            colors[node] = cor

    # Desenhar o grafo colorido
    pos = nx.spring_layout(G_temp, seed=42)
    plt.figure(figsize=(12, 10))
    nx.draw(
        G_temp, pos,
        with_labels=False,
        node_color=[colors[n] for n in G_temp.nodes()],
        node_size=50
    )
    labels = {n: G_temp.nodes[n]["partido"] for n in G_temp.nodes()}
    nx.draw_networkx_labels(G_temp, pos, labels, font_size=6)
    plt.title("Grafo de Clusterização (sem heatmap)")
    plt.tight_layout()
    plt.show()

def clusterizacao_com_heatmap_girvan_newman():
    """
    Gera o grafo de clusterização com heatmap de fundo mostrando concentração
    de partidos de esquerda (vermelho) e direita (azul).
    """
    isolates = list(nx.isolates(G))
    G_temp = G.copy()
    G_temp.remove_nodes_from(isolates)
    comp = nx.algorithms.community.girvan_newman(G_temp)

    # Obter as primeiras divisões do grafo
    first_level = next(comp)      # Primeira divisão (2 comunidades)
    second_level = next(comp)     # Segunda divisão (mais fragmentada)

    communities = [list(c) for c in second_level]

    # Define as cores específicas para cada comunidade
    cores_comunidades = ['yellow', 'green', 'blue', 'purple', 'pink']
    
    # Atribuir uma cor a cada comunidade
    colors = {}
    for i, comm in enumerate(communities):
        cor = cores_comunidades[i % len(cores_comunidades)]
        for node in comm:
            colors[node] = cor

    # Calcular posições dos nós
    pos = nx.spring_layout(G_temp, seed=42)
    
    # Preparar dados para o heatmap
    # Extrair coordenadas e orientações políticas
    x_coords = []
    y_coords = []
    orientacoes = []
    
    for node in G_temp.nodes():
        x, y = pos[node]
        x_coords.append(x)
        y_coords.append(y)
        partido = G_temp.nodes[node].get('partido', '')
        orientacao = obter_orientacao_politica(partido)
        orientacoes.append(orientacao)
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    orientacoes = np.array(orientacoes)
    
    # Criar grid para o heatmap
    x_min, x_max = x_coords.min() - 0.1, x_coords.max() + 0.1
    y_min, y_max = y_coords.min() - 0.1, y_coords.max() + 0.1
    
    # Resolução do grid
    resolution = 100
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Calcular densidade ponderada por orientação política
    # Usar Kernel Density Estimation simples (Gaussiano)
    heatmap_values = np.zeros_like(xi_grid)
    
    for i in range(len(x_coords)):
        x_node, y_node = x_coords[i], y_coords[i]
        orientacao = orientacoes[i]
        
        # Calcular distância de cada ponto do grid até este nó
        dist_x = xi_grid - x_node
        dist_y = yi_grid - y_node
        dist_sq = dist_x**2 + dist_y**2
        
        # Kernel gaussiano (bandwidth ajustável)
        bandwidth = 0.15
        kernel = np.exp(-dist_sq / (2 * bandwidth**2))
        
        # Adicionar contribuição ponderada pela orientação
        # -1 (esquerda) contribui negativamente (vermelho)
        # 0 (centro) não contribui
        # 1 (direita) contribui positivamente (azul)
        heatmap_values += kernel * orientacao
    
    # Normalizar o heatmap
    if heatmap_values.max() > 0 or heatmap_values.min() < 0:
        max_abs = max(abs(heatmap_values.max()), abs(heatmap_values.min()))
        if max_abs > 0:
            heatmap_values = heatmap_values / max_abs
    
    # Criar figura
    plt.figure(figsize=(12, 10))
    
    # Desenhar heatmap
    # Usar colormap customizado: vermelho para esquerda, amarelo para centro, azul para direita
    from matplotlib.colors import LinearSegmentedColormap
    
    # Criar colormap customizado (vermelho -> amarelo -> azul)
    # Valores negativos (esquerda) = vermelho
    # Valores próximos de 0 (centro) = amarelo
    # Valores positivos (direita) = azul
    colors_cmap = ['red', 'yellow', 'blue']
    n_bins = 100
    cmap_custom = LinearSegmentedColormap.from_list('custom', colors_cmap, N=n_bins)
    
    # Plotar heatmap
    im = plt.contourf(xi_grid, yi_grid, heatmap_values, levels=50, cmap=cmap_custom, alpha=0.6, vmin=-1, vmax=1)
    plt.colorbar(im, label='Orientação Política (Vermelho=Esquerda, Amarelo=Centro, Azul=Direita)', shrink=0.8)
    
    # Desenhar o grafo por cima
    nx.draw(
        G_temp, pos,
        with_labels=False,
        node_color=[colors[n] for n in G_temp.nodes()],
        node_size=50,
        edgecolors='black',
        linewidths=0.5
    )
    labels = {n: G_temp.nodes[n]["partido"] for n in G_temp.nodes()}
    nx.draw_networkx_labels(G_temp, pos, labels, font_size=6)
    
    plt.title("Grafo de Clusterização com Heatmap de Orientação Política")
    plt.tight_layout()
    plt.show()

def clusterizacao_sem_heatmap_ravasz():
    """
    Gera o grafo de clusterização sem heatmap de fundo usando o algoritmo Ravasz
    (agrupamento hierárquico baseado em similaridade).
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    isolates = list(nx.isolates(G))
    G_temp = G.copy()
    G_temp.remove_nodes_from(isolates)
    
    # Obter lista de nós ordenada
    nodes = list(G_temp.nodes())
    n_nodes = len(nodes)
    
    # Calcular matriz de distância baseada em caminho mais curto
    # O algoritmo Ravasz usa similaridade, então vamos usar distância inversa
    distance_matrix = np.zeros((n_nodes, n_nodes))
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                try:
                    # Calcula o caminho mais curto
                    path_length = nx.shortest_path_length(G_temp, node1, node2)
                    # Converte para distância (quanto maior o caminho, maior a distância)
                    distance_matrix[i, j] = path_length
                except nx.NetworkXNoPath:
                    # Se não há caminho, distância é muito grande
                    distance_matrix[i, j] = n_nodes
    
    # Converter matriz de distância para formato condensado (necessário para linkage)
    condensed_distances = squareform(distance_matrix)
    
    # Aplicar agrupamento hierárquico (linkage)
    # Usando método 'average' (UPGMA) que é comum no algoritmo Ravasz
    Z = linkage(condensed_distances, method='average')
    
    # Determinar número de comunidades (similar ao segundo nível do Girvan-Newman)
    # Vamos usar um threshold que produza um número razoável de comunidades
    # Ou podemos usar maxclust para especificar número máximo de clusters
    num_communities = min(5, n_nodes // 2)  # Limita a 5 comunidades ou metade dos nós
    if num_communities < 2:
        num_communities = 2
    
    # Obter labels de cluster
    cluster_labels = fcluster(Z, num_communities, criterion='maxclust')
    
    # Agrupar nós por comunidade
    communities = {}
    for i, node in enumerate(nodes):
        cluster_id = cluster_labels[i]
        if cluster_id not in communities:
            communities[cluster_id] = []
        communities[cluster_id].append(node)
    
    communities_list = [list(c) for c in communities.values()]
    
    # Define as cores específicas para cada comunidade
    cores_comunidades = ['yellow', 'green', 'blue', 'purple', 'pink']
    
    # Atribuir uma cor a cada comunidade
    colors = {}
    for i, comm in enumerate(communities_list):
        cor = cores_comunidades[i % len(cores_comunidades)]
        for node in comm:
            colors[node] = cor
    
    # Desenhar o grafo colorido
    pos = nx.spring_layout(G_temp, seed=42)
    plt.figure(figsize=(12, 10))
    nx.draw(
        G_temp, pos,
        with_labels=False,
        node_color=[colors[n] for n in G_temp.nodes()],
        node_size=50
    )
    labels = {n: G_temp.nodes[n]["partido"] for n in G_temp.nodes()}
    nx.draw_networkx_labels(G_temp, pos, labels, font_size=6)
    plt.title("Grafo de Clusterização - Ravasz Algorithm (sem heatmap)")
    plt.tight_layout()
    plt.show()

def clusterizacao_com_heatmap_ravasz():
    """
    Gera o grafo de clusterização com heatmap de fundo usando o algoritmo Ravasz
    mostrando concentração de partidos de esquerda (vermelho) e direita (azul).
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    isolates = list(nx.isolates(G))
    G_temp = G.copy()
    G_temp.remove_nodes_from(isolates)
    
    # Obter lista de nós ordenada
    nodes = list(G_temp.nodes())
    n_nodes = len(nodes)
    
    # Calcular matriz de distância baseada em caminho mais curto
    distance_matrix = np.zeros((n_nodes, n_nodes))
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                try:
                    path_length = nx.shortest_path_length(G_temp, node1, node2)
                    distance_matrix[i, j] = path_length
                except nx.NetworkXNoPath:
                    distance_matrix[i, j] = n_nodes
    
    # Converter matriz de distância para formato condensado
    condensed_distances = squareform(distance_matrix)
    
    # Aplicar agrupamento hierárquico
    Z = linkage(condensed_distances, method='average')
    
    # Determinar número de comunidades
    num_communities = min(5, n_nodes // 2)
    if num_communities < 2:
        num_communities = 2
    
    # Obter labels de cluster
    cluster_labels = fcluster(Z, num_communities, criterion='maxclust')
    
    # Agrupar nós por comunidade
    communities = {}
    for i, node in enumerate(nodes):
        cluster_id = cluster_labels[i]
        if cluster_id not in communities:
            communities[cluster_id] = []
        communities[cluster_id].append(node)
    
    communities_list = [list(c) for c in communities.values()]
    
    # Define as cores específicas para cada comunidade
    cores_comunidades = ['yellow', 'green', 'blue', 'purple', 'pink']
    
    # Atribuir uma cor a cada comunidade
    colors = {}
    for i, comm in enumerate(communities_list):
        cor = cores_comunidades[i % len(cores_comunidades)]
        for node in comm:
            colors[node] = cor
    
    # Calcular posições dos nós
    pos = nx.spring_layout(G_temp, seed=42)
    
    # Preparar dados para o heatmap
    x_coords = []
    y_coords = []
    orientacoes = []
    
    for node in G_temp.nodes():
        x, y = pos[node]
        x_coords.append(x)
        y_coords.append(y)
        partido = G_temp.nodes[node].get('partido', '')
        orientacao = obter_orientacao_politica(partido)
        orientacoes.append(orientacao)
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    orientacoes = np.array(orientacoes)
    
    # Criar grid para o heatmap
    x_min, x_max = x_coords.min() - 0.1, x_coords.max() + 0.1
    y_min, y_max = y_coords.min() - 0.1, y_coords.max() + 0.1
    
    # Resolução do grid
    resolution = 100
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Calcular densidade ponderada por orientação política
    heatmap_values = np.zeros_like(xi_grid)
    
    for i in range(len(x_coords)):
        x_node, y_node = x_coords[i], y_coords[i]
        orientacao = orientacoes[i]
        
        dist_x = xi_grid - x_node
        dist_y = yi_grid - y_node
        dist_sq = dist_x**2 + dist_y**2
        
        bandwidth = 0.15
        kernel = np.exp(-dist_sq / (2 * bandwidth**2))
        heatmap_values += kernel * orientacao
    
    # Normalizar o heatmap
    if heatmap_values.max() > 0 or heatmap_values.min() < 0:
        max_abs = max(abs(heatmap_values.max()), abs(heatmap_values.min()))
        if max_abs > 0:
            heatmap_values = heatmap_values / max_abs
    
    # Criar figura
    plt.figure(figsize=(12, 10))
    
    # Desenhar heatmap
    from matplotlib.colors import LinearSegmentedColormap
    
    colors_cmap = ['red', 'yellow', 'blue']
    n_bins = 100
    cmap_custom = LinearSegmentedColormap.from_list('custom', colors_cmap, N=n_bins)
    
    # Plotar heatmap
    im = plt.contourf(xi_grid, yi_grid, heatmap_values, levels=50, cmap=cmap_custom, alpha=0.6, vmin=-1, vmax=1)
    plt.colorbar(im, label='Orientação Política (Vermelho=Esquerda, Amarelo=Centro, Azul=Direita)', shrink=0.8)
    
    # Desenhar o grafo por cima
    nx.draw(
        G_temp, pos,
        with_labels=False,
        node_color=[colors[n] for n in G_temp.nodes()],
        node_size=50,
        edgecolors='black',
        linewidths=0.5
    )
    labels = {n: G_temp.nodes[n]["partido"] for n in G_temp.nodes()}
    nx.draw_networkx_labels(G_temp, pos, labels, font_size=6)
    
    plt.title("Grafo de Clusterização - Ravasz Algorithm com Heatmap de Orientação Política")
    plt.tight_layout()
    plt.show()

# ============================================================================
# MÉTRICAS DE NÍVEL DE REDE (GRAFO COMPLETO)
# ============================================================================

def calcular_metricas_nivel_rede(grafo=None):
    """
    Calcula e exibe métricas de nível de rede (grafo completo).
    
    Args:
        grafo: Grafo NetworkX (padrão: G global)
    """
    if grafo is None:
        grafo = G
    
    print("=" * 80)
    print("MÉTRICAS DE NÍVEL DE REDE")
    print("=" * 80)
    
    # Informações básicas
    n_nodes = grafo.number_of_nodes()
    n_edges = grafo.number_of_edges()
    print(f"\nNúmero de nós (deputados): {n_nodes}")
    print(f"Número de arestas (conexões): {n_edges}")
    
    # 1. Densidade
    density = nx.density(grafo)
    print(f"\n1. DENSIDADE DA REDE: {density:.6f}")
    print(f"   (Proporção de arestas existentes vs. possíveis)")
    print(f"   Arestas possíveis: {n_nodes * (n_nodes - 1) / 2}")
    print(f"   Arestas existentes: {n_edges}")
    
    # 2. Coeficiente de agrupamento
    clustering_global = nx.transitivity(grafo)  # Coeficiente global
    clustering_medio = nx.average_clustering(grafo)  # Média dos coeficientes locais
    print(f"\n2. COEFICIENTE DE AGRUPAMENTO:")
    print(f"   Global (transitividade): {clustering_global:.6f}")
    print(f"   Médio: {clustering_medio:.6f}")
    
    # 3. Diâmetro e raio (apenas para componente maior)
    if nx.is_connected(grafo):
        diameter = nx.diameter(grafo)
        radius = nx.radius(grafo)
        avg_path_length = nx.average_shortest_path_length(grafo)
        print(f"\n3. DISTÂNCIAS:")
        print(f"   Diâmetro: {diameter}")
        print(f"   Raio: {radius}")
        print(f"   Caminho médio: {avg_path_length:.4f}")
    else:
        # Para grafos desconectados, calcular para maior componente
        largest_cc = max(nx.connected_components(grafo), key=len)
        G_largest = grafo.subgraph(largest_cc)
        if len(largest_cc) > 1:
            diameter = nx.diameter(G_largest)
            radius = nx.radius(G_largest)
            avg_path_length = nx.average_shortest_path_length(G_largest)
            print(f"\n3. DISTÂNCIAS (maior componente conexa, {len(largest_cc)} nós):")
            print(f"   Diâmetro: {diameter}")
            print(f"   Raio: {radius}")
            print(f"   Caminho médio: {avg_path_length:.4f}")
        else:
            print(f"\n3. DISTÂNCIAS: Grafo desconectado, maior componente tem apenas 1 nó")
    
    # 4. Assortatividade
    print(f"\n4. ASSORTATIVIDADE:")
    
    # Por partido
    try:
        partido_dict = {n: grafo.nodes[n].get('partido', 'Sem partido') for n in grafo.nodes()}
        assort_partido = nx.attribute_assortativity_coefficient(grafo, 'partido')
        print(f"   Por partido: {assort_partido:.6f}")
    except:
        print(f"   Por partido: Não calculável (verificar atributos)")
    
    # Por UF
    try:
        uf_dict = {n: grafo.nodes[n].get('uf', 'Sem UF') for n in grafo.nodes()}
        assort_uf = nx.attribute_assortativity_coefficient(grafo, 'uf')
        print(f"   Por UF: {assort_uf:.6f}")
    except:
        print(f"   Por UF: Não calculável")
    
    # Por grau (assortatividade de grau)
    try:
        assort_degree = nx.degree_assortativity_coefficient(grafo)
        print(f"   Por grau: {assort_degree:.6f}")
    except:
        print(f"   Por grau: Não calculável")
    
    # 5. Componentes conexas
    components = list(nx.connected_components(grafo))
    n_components = len(components)
    sizes = [len(c) for c in components]
    largest_component = max(sizes) if sizes else 0
    
    print(f"\n5. COMPONENTES CONEXAS:")
    print(f"   Número de componentes: {n_components}")
    print(f"   Tamanho da maior componente: {largest_component} nós")
    print(f"   Tamanhos das componentes: {sorted(sizes, reverse=True)[:10]}")  # Top 10
    
    # 6. Análise de isolados
    isolates = list(nx.isolates(grafo))
    print(f"\n6. NÓS ISOLADOS: {len(isolates)}")
    
    print("\n" + "=" * 80)

def calcular_modularidade_comunidades(grafo, comunidades, metodo_nome="Método"):
    """
    Calcula a modularidade de um conjunto de comunidades.
    
    Args:
        grafo: Grafo NetworkX
        comunidades: Lista de listas de nós (cada lista é uma comunidade)
        metodo_nome: Nome do método usado (para exibição)
    
    Returns:
        float: Valor da modularidade
    """
    # Converter comunidades para formato NetworkX
    communities_dict = {}
    for i, comm in enumerate(comunidades):
        for node in comm:
            communities_dict[node] = i
    
    # Calcular modularidade
    modularity = nx.community.modularity(grafo, [set(c) for c in comunidades])
    
    print(f"\n{metodo_nome}:")
    print(f"  Número de comunidades: {len(comunidades)}")
    print(f"  Tamanhos das comunidades: {[len(c) for c in comunidades]}")
    print(f"  Modularidade: {modularity:.6f}")
    
    return modularity

# ============================================================================
# MÉTRICAS DE NÍVEL DE NÓ (DEPUTADOS)
# ============================================================================

def calcular_metricas_nivel_no(grafo=None, top_n=10):
    """
    Calcula e exibe métricas de nível de nó (deputados).
    
    Args:
        grafo: Grafo NetworkX (padrão: G global)
        top_n: Número de top deputados para exibir em cada métrica
    """
    if grafo is None:
        grafo = G
    
    print("=" * 80)
    print("MÉTRICAS DE NÍVEL DE NÓ (DEPUTADOS)")
    print("=" * 80)
    
    # Remover isolados para algumas métricas
    isolates = list(nx.isolates(grafo))
    G_temp = grafo.copy()
    if isolates:
        G_temp.remove_nodes_from(isolates)
        print(f"\nNota: {len(isolates)} nós isolados foram removidos para algumas métricas")
    
    # 1. Centralidade de Grau (Degree Centrality)
    print(f"\n1. CENTRALIDADE DE GRAU (Top {top_n}):")
    degree_centrality = nx.degree_centrality(G_temp)
    sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for i, (node, centrality) in enumerate(sorted_degree, 1):
        nome = G_temp.nodes[node].get('nome', 'N/A')
        partido = G_temp.nodes[node].get('partido', 'N/A')
        grau = G_temp.degree(node)
        print(f"   {i}. {nome} ({partido}) - Centralidade: {centrality:.6f}, Grau: {grau}")
    
    # 2. Centralidade de Intermediação (Betweenness Centrality)
    print(f"\n2. CENTRALIDADE DE INTERMEDIAÇÃO (Top {top_n}):")
    betweenness = nx.betweenness_centrality(G_temp)
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for i, (node, centrality) in enumerate(sorted_betweenness, 1):
        nome = G_temp.nodes[node].get('nome', 'N/A')
        partido = G_temp.nodes[node].get('partido', 'N/A')
        print(f"   {i}. {nome} ({partido}) - Centralidade: {centrality:.6f}")
    
    # 3. Centralidade de Proximidade (Closeness Centrality)
    closeness = {}
    if nx.is_connected(G_temp) or len(max(nx.connected_components(G_temp), key=len)) > 1:
        print(f"\n3. CENTRALIDADE DE PROXIMIDADE (Top {top_n}):")
        closeness = nx.closeness_centrality(G_temp)
        sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for i, (node, centrality) in enumerate(sorted_closeness, 1):
            nome = G_temp.nodes[node].get('nome', 'N/A')
            partido = G_temp.nodes[node].get('partido', 'N/A')
            print(f"   {i}. {nome} ({partido}) - Centralidade: {centrality:.6f}")
    else:
        print(f"\n3. CENTRALIDADE DE PROXIMIDADE: Não calculável (grafo muito desconectado)")
    
    # 4. Centralidade de Autovetor (Eigenvector Centrality)
    try:
        print(f"\n4. CENTRALIDADE DE AUTOVETOR (Top {top_n}):")
        eigenvector = nx.eigenvector_centrality(G_temp, max_iter=1000)
        sorted_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for i, (node, centrality) in enumerate(sorted_eigenvector, 1):
            nome = G_temp.nodes[node].get('nome', 'N/A')
            partido = G_temp.nodes[node].get('partido', 'N/A')
            print(f"   {i}. {nome} ({partido}) - Centralidade: {centrality:.6f}")
    except:
        print(f"\n4. CENTRALIDADE DE AUTOVETOR: Não calculável")
    
    # 5. PageRank
    print(f"\n5. PAGERANK (Top {top_n}):")
    pagerank = nx.pagerank(G_temp)
    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for i, (node, centrality) in enumerate(sorted_pagerank, 1):
        nome = G_temp.nodes[node].get('nome', 'N/A')
        partido = G_temp.nodes[node].get('partido', 'N/A')
        print(f"   {i}. {nome} ({partido}) - PageRank: {centrality:.6f}")
    
    # 6. Coeficiente de Agrupamento Local
    print(f"\n6. COEFICIENTE DE AGRUPAMENTO LOCAL (Top {top_n} - maior agrupamento):")
    clustering = nx.clustering(G_temp)
    sorted_clustering = sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for i, (node, coef) in enumerate(sorted_clustering, 1):
        nome = G_temp.nodes[node].get('nome', 'N/A')
        partido = G_temp.nodes[node].get('partido', 'N/A')
        print(f"   {i}. {nome} ({partido}) - Coeficiente: {coef:.6f}")
    
    # 7. K-Core (Coreness)
    print(f"\n7. K-CORE (Coreness - Top {top_n}):")
    try:
        core_number = nx.core_number(G_temp)
        sorted_core = sorted(core_number.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for i, (node, core) in enumerate(sorted_core, 1):
            nome = G_temp.nodes[node].get('nome', 'N/A')
            partido = G_temp.nodes[node].get('partido', 'N/A')
            print(f"   {i}. {nome} ({partido}) - K-Core: {core}")
    except:
        print(f"   Não calculável")
    
    print("\n" + "=" * 80)
    
    # Retornar dicionários com todas as métricas para uso posterior
    return {
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness,
        'closeness_centrality': closeness,
        'eigenvector_centrality': eigenvector if 'eigenvector' in locals() else {},
        'pagerank': pagerank,
        'clustering': clustering,
        'core_number': core_number if 'core_number' in locals() else {}
    }

def visualizar_centralidades(grafo=None, metricas=None):
    """
    Visualiza as centralidades dos nós no grafo.
    
    Args:
        grafo: Grafo NetworkX (padrão: G global)
        metricas: Dicionário com métricas calculadas (opcional)
    """
    if grafo is None:
        grafo = G
    
    if metricas is None:
        metricas = calcular_metricas_nivel_no(grafo, top_n=5)
    
    isolates = list(nx.isolates(grafo))
    G_temp = grafo.copy()
    if isolates:
        G_temp.remove_nodes_from(isolates)
    
    pos = nx.spring_layout(G_temp, seed=42)
    
    # Criar subplots para diferentes centralidades
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Degree Centrality
    if 'degree_centrality' in metricas:
        node_sizes = [metricas['degree_centrality'].get(n, 0) * 2000 for n in G_temp.nodes()]
        nx.draw(G_temp, pos, ax=axes[0], node_size=node_sizes, node_color='lightblue', 
                with_labels=False, edge_color='gray', alpha=0.7)
        axes[0].set_title('Degree Centrality', fontsize=12, fontweight='bold')
    
    # 2. Betweenness Centrality
    if 'betweenness_centrality' in metricas:
        node_sizes = [metricas['betweenness_centrality'].get(n, 0) * 5000 for n in G_temp.nodes()]
        nx.draw(G_temp, pos, ax=axes[1], node_size=node_sizes, node_color='lightcoral', 
                with_labels=False, edge_color='gray', alpha=0.7)
        axes[1].set_title('Betweenness Centrality', fontsize=12, fontweight='bold')
    
    # 3. Closeness Centrality
    if 'closeness_centrality' in metricas and metricas['closeness_centrality']:
        node_sizes = [metricas['closeness_centrality'].get(n, 0) * 2000 for n in G_temp.nodes()]
        nx.draw(G_temp, pos, ax=axes[2], node_size=node_sizes, node_color='lightgreen', 
                with_labels=False, edge_color='gray', alpha=0.7)
        axes[2].set_title('Closeness Centrality', fontsize=12, fontweight='bold')
    
    # 4. Eigenvector Centrality
    if 'eigenvector_centrality' in metricas and metricas['eigenvector_centrality']:
        node_sizes = [metricas['eigenvector_centrality'].get(n, 0) * 3000 for n in G_temp.nodes()]
        nx.draw(G_temp, pos, ax=axes[3], node_size=node_sizes, node_color='plum', 
                with_labels=False, edge_color='gray', alpha=0.7)
        axes[3].set_title('Eigenvector Centrality', fontsize=12, fontweight='bold')
    
    # 5. PageRank
    if 'pagerank' in metricas:
        node_sizes = [metricas['pagerank'].get(n, 0) * 10000 for n in G_temp.nodes()]
        nx.draw(G_temp, pos, ax=axes[4], node_size=node_sizes, node_color='wheat', 
                with_labels=False, edge_color='gray', alpha=0.7)
        axes[4].set_title('PageRank', fontsize=12, fontweight='bold')
    
    # 6. Clustering Coefficient
    if 'clustering' in metricas:
        node_sizes = [metricas['clustering'].get(n, 0) * 2000 + 50 for n in G_temp.nodes()]
        nx.draw(G_temp, pos, ax=axes[5], node_size=node_sizes, node_color='lightcyan', 
                with_labels=False, edge_color='gray', alpha=0.7)
        axes[5].set_title('Clustering Coefficient', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('Visualização de Centralidades e Métricas de Nó', fontsize=16, fontweight='bold', y=1.02)
    plt.show()

# ============================================================================
# ANÁLISE DE COMUNIDADES
# ============================================================================

def obter_comunidades_girvan_newman(grafo=None, nivel=2):
    """
    Obtém comunidades usando algoritmo Girvan-Newman.
    
    Args:
        grafo: Grafo NetworkX (padrão: G global)
        nivel: Nível de divisão (1 = primeira divisão, 2 = segunda, etc.)
    
    Returns:
        list: Lista de comunidades (cada comunidade é uma lista de nós)
    """
    if grafo is None:
        grafo = G
    
    isolates = list(nx.isolates(grafo))
    G_temp = grafo.copy()
    G_temp.remove_nodes_from(isolates)
    
    comp = nx.algorithms.community.girvan_newman(G_temp)
    
    # Avançar até o nível desejado
    comunidades = None
    for i in range(nivel):
        comunidades = next(comp)
    
    return [list(c) for c in comunidades]

def obter_comunidades_ravasz(grafo=None, num_communities=None):
    """
    Obtém comunidades usando algoritmo Ravasz (hierarchical clustering).
    
    Args:
        grafo: Grafo NetworkX (padrão: G global)
        num_communities: Número de comunidades desejadas (None = automático)
    
    Returns:
        list: Lista de comunidades (cada comunidade é uma lista de nós)
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    if grafo is None:
        grafo = G
    
    isolates = list(nx.isolates(grafo))
    G_temp = grafo.copy()
    G_temp.remove_nodes_from(isolates)
    
    nodes = list(G_temp.nodes())
    n_nodes = len(nodes)
    
    if n_nodes < 2:
        return [[n] for n in nodes]
    
    # Calcular matriz de distância
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                try:
                    path_length = nx.shortest_path_length(G_temp, node1, node2)
                    distance_matrix[i, j] = path_length
                except nx.NetworkXNoPath:
                    distance_matrix[i, j] = n_nodes
    
    condensed_distances = squareform(distance_matrix)
    Z = linkage(condensed_distances, method='average')
    
    if num_communities is None:
        num_communities = min(5, n_nodes // 2)
        if num_communities < 2:
            num_communities = 2
    
    cluster_labels = fcluster(Z, num_communities, criterion='maxclust')
    
    communities = {}
    for i, node in enumerate(nodes):
        cluster_id = cluster_labels[i]
        if cluster_id not in communities:
            communities[cluster_id] = []
        communities[cluster_id].append(node)
    
    return [list(c) for c in communities.values()]

def obter_comunidades_louvain(grafo=None):
    """
    Obtém comunidades usando algoritmo Louvain.
    
    Args:
        grafo: Grafo NetworkX (padrão: G global)
    
    Returns:
        list: Lista de comunidades (cada comunidade é uma lista de nós)
    """
    if grafo is None:
        grafo = G
    
    isolates = list(nx.isolates(grafo))
    G_temp = grafo.copy()
    G_temp.remove_nodes_from(isolates)
    
    try:
        communities = nx.community.louvain_communities(G_temp, seed=42)
        return [list(c) for c in communities]
    except:
        # Fallback se não disponível
        return [[n] for n in G_temp.nodes()]

def analisar_comunidades(grafo=None):
    """
    Realiza análise comparativa de comunidades usando diferentes algoritmos.
    
    Args:
        grafo: Grafo NetworkX (padrão: G global)
    """
    if grafo is None:
        grafo = G
    
    print("=" * 80)
    print("ANÁLISE DE COMUNIDADES")
    print("=" * 80)
    
    isolates = list(nx.isolates(grafo))
    G_temp = grafo.copy()
    G_temp.remove_nodes_from(isolates)
    
    # 1. Girvan-Newman
    print("\n1. ALGORITMO GIRVAN-NEWMAN:")
    comunidades_gn = obter_comunidades_girvan_newman(G_temp, nivel=2)
    modularity_gn = calcular_modularidade_comunidades(G_temp, comunidades_gn, "Girvan-Newman")
    
    # 2. Ravasz
    print("\n2. ALGORITMO RAVASZ:")
    comunidades_ravasz = obter_comunidades_ravasz(G_temp)
    modularity_ravasz = calcular_modularidade_comunidades(G_temp, comunidades_ravasz, "Ravasz")
    
    # 3. Louvain
    print("\n3. ALGORITMO LOUVAIN:")
    comunidades_louvain = obter_comunidades_louvain(G_temp)
    modularity_louvain = calcular_modularidade_comunidades(G_temp, comunidades_louvain, "Louvain")
    
    # Comparação
    print("\n" + "-" * 80)
    print("COMPARAÇÃO DOS ALGORITMOS:")
    print("-" * 80)
    print(f"Girvan-Newman: {len(comunidades_gn)} comunidades, Modularidade: {modularity_gn:.6f}")
    print(f"Ravasz:        {len(comunidades_ravasz)} comunidades, Modularidade: {modularity_ravasz:.6f}")
    print(f"Louvain:       {len(comunidades_louvain)} comunidades, Modularidade: {modularity_louvain:.6f}")
    
    melhor = max([
        ("Girvan-Newman", modularity_gn),
        ("Ravasz", modularity_ravasz),
        ("Louvain", modularity_louvain)
    ], key=lambda x: x[1])
    print(f"\nMelhor modularidade: {melhor[0]} ({melhor[1]:.6f})")
    
    # Análise de composição política das comunidades
    print("\n" + "-" * 80)
    print("COMPOSIÇÃO POLÍTICA DAS COMUNIDADES (Girvan-Newman):")
    print("-" * 80)
    analisar_composicao_politica_comunidades(G_temp, comunidades_gn)
    
    print("\n" + "=" * 80)
    
    return {
        'girvan_newman': comunidades_gn,
        'ravasz': comunidades_ravasz,
        'louvain': comunidades_louvain,
        'modularities': {
            'girvan_newman': modularity_gn,
            'ravasz': modularity_ravasz,
            'louvain': modularity_louvain
        }
    }

def analisar_composicao_politica_comunidades(grafo, comunidades):
    """
    Analisa a composição política (partidos e orientações) de cada comunidade.
    
    Args:
        grafo: Grafo NetworkX
        comunidades: Lista de comunidades
    """
    for i, comm in enumerate(comunidades, 1):
        print(f"\nComunidade {i} ({len(comm)} deputados):")
        
        # Contar partidos
        partidos = {}
        orientacoes = {'Extrema-esquerda':0, 
                       'Esquerda': 0, 
                       'Centro-esquerda':0, 
                       'Centro': 0, 
                       'Centro-direita': 0,
                       'Direita': 0, 
                       'Extrema-direita':0,
                       'Sem classificação': 0}
        
        for node in comm:
            partido = grafo.nodes[node].get('partido', 'Sem partido')
            partidos[partido] = partidos.get(partido, 0) + 1
            
            # Determinar orientação
            orientacao = obter_orientacao_politica(partido)
            if orientacao == -1:
                orientacoes['Extrema-esquerda'] += 1
            elif orientacao ==  -0.6:
                orientacoes['Esquerda'] += 1
            elif orientacao == -0.3:
                orientacoes['Centro-esquerda'] += 1
            elif orientacao == 0:
                orientacoes['Centro'] += 1
            elif orientacao == 0.3:
                orientacoes['Centro-direita'] += 1
            elif orientacao ==  0.6:
                orientacoes['Direita'] += 1
            elif orientacao == 1:
                orientacoes['Extrema-direita'] += 1
            else:
                orientacoes['Sem classificação'] += 1 
        
        # Partidos mais frequentes
        partidos_sorted = sorted(partidos.items(), key=lambda x: x[1], reverse=True)
        print(f"  Partidos: {dict(partidos_sorted[:5])}")  # Top 5
        
        # Orientação política
        print(f"  Orientação política: {orientacoes}")

def visualizar_comparacao_comunidades(grafo=None):
    """
    Visualiza comparação entre diferentes algoritmos de detecção de comunidades.
    
    Args:
        grafo: Grafo NetworkX (padrão: G global)
    """
    if grafo is None:
        grafo = G
    
    isolates = list(nx.isolates(grafo))
    G_temp = grafo.copy()
    G_temp.remove_nodes_from(isolates)
    
    comunidades_gn = obter_comunidades_girvan_newman(G_temp, nivel=2)
    comunidades_ravasz = obter_comunidades_ravasz(G_temp)
    comunidades_louvain = obter_comunidades_louvain(G_temp)
    
    pos = nx.spring_layout(G_temp, seed=42)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    cores_comunidades = ['yellow', 'green', 'blue', 'purple', 'pink', 'orange', 'cyan', 'magenta']
    
    # Girvan-Newman
    colors_gn = {}
    for i, comm in enumerate(comunidades_gn):
        cor = cores_comunidades[i % len(cores_comunidades)]
        for node in comm:
            colors_gn[node] = cor
    
    nx.draw(G_temp, pos, ax=axes[0], node_color=[colors_gn.get(n, 'gray') for n in G_temp.nodes()],
            node_size=50, with_labels=False, edge_color='gray', alpha=0.7)
    axes[0].set_title(f'Girvan-Newman\n({len(comunidades_gn)} comunidades)', fontweight='bold')
    
    # Ravasz
    colors_ravasz = {}
    for i, comm in enumerate(comunidades_ravasz):
        cor = cores_comunidades[i % len(cores_comunidades)]
        for node in comm:
            colors_ravasz[node] = cor
    
    nx.draw(G_temp, pos, ax=axes[1], node_color=[colors_ravasz.get(n, 'gray') for n in G_temp.nodes()],
            node_size=50, with_labels=False, edge_color='gray', alpha=0.7)
    axes[1].set_title(f'Ravasz\n({len(comunidades_ravasz)} comunidades)', fontweight='bold')
    
    # Louvain
    colors_louvain = {}
    for i, comm in enumerate(comunidades_louvain):
        cor = cores_comunidades[i % len(cores_comunidades)]
        for node in comm:
            colors_louvain[node] = cor
    
    nx.draw(G_temp, pos, ax=axes[2], node_color=[colors_louvain.get(n, 'gray') for n in G_temp.nodes()],
            node_size=50, with_labels=False, edge_color='gray', alpha=0.7)
    axes[2].set_title(f'Louvain\n({len(comunidades_louvain)} comunidades)', fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('Comparação de Algoritmos de Detecção de Comunidades', fontsize=14, fontweight='bold', y=1.02)
    plt.show()

def plotarDistribuicaoDeGrau():
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)

    print("Grau -> Frequência")
    for grau, freq in sorted(degree_count.items()):
        print(f"{grau} -> {freq}")

    deg, cnt = zip(*degree_count.items())

    plt.figure(figsize=(8, 5))
    plt.bar(deg, cnt)
    plt.title("Distribuição de grau")
    plt.xlabel("Grau")
    plt.ylabel("Número de nós")
    plt.show()


def criarRede():
    contar_linhas_por_periodo()
    acharVotacoes()
    criarVertices()
    calcularArestas()
    #calcularArestasV2()
    # Usa GraphML para preservar todos os nós (incluindo isolados) e pesos das arestas
    nx.write_graphml(G, "grafoSimilaridade.graphml")


G = nx.read_graphml("grafoSimilaridade1.graphml")
pos = nx.spring_layout(G, k=0.8, iterations=200)
labels = {n: G.nodes[n]["partido"] for n in G.nodes()}

#plota sem as cores de orientação política
plt.figure(figsize=(10, 10))
nx.draw(G, pos, 
        with_labels=False, 
        node_size=50)
nx.draw_networkx_labels(G, pos, labels)
plt.show()

plotarDistribuicaoDeGrau()

#plota com as cores
cores_vertices = pintarVerticesPorOrientacaoPolitica()
plt.figure(figsize=(10, 10))
nx.draw(G, pos, 
        with_labels=False, 
        node_size=50, 
        node_color=cores_vertices)
nx.draw_networkx_labels(G, pos, labels)

# Adiciona legenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='Esquerda (Extrema-esquerda, Esquerda, Centro-esquerda)'),
    Patch(facecolor='yellow', label='Centro'),
    Patch(facecolor='blue', label='Direita (Centro-direita, Direita, Extrema-direita)'),
    Patch(facecolor='gray', label='Sem classificação')
]
plt.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

plt.show()

components = nx.connected_components(G)
sizes = [len(c) for c in components]
print(sizes)
print(G.number_of_edges())

# Gerar grafo sem heatmap primeiro
print("Gerando grafo sem heatmap...")
clusterizacao_sem_heatmap_girvan_newman()

# Gerar grafo com heatmap
print("Gerando grafo com heatmap...")
clusterizacao_com_heatmap_girvan_newman()

clusterizacao_sem_heatmap_ravasz()
clusterizacao_com_heatmap_ravasz()

# ============================================================================
# EXEMPLOS DE USO DAS MÉTRICAS
# ============================================================================

# Descomente as linhas abaixo para executar as análises:

# 1. Métricas de nível de rede
print("\n" + "="*80)
print("CALCULANDO MÉTRICAS DE NÍVEL DE REDE...")
print("="*80)
calcular_metricas_nivel_rede()

# 2. Métricas de nível de nó
print("\n" + "="*80)
print("CALCULANDO MÉTRICAS DE NÍVEL DE NÓ...")
print("="*80)
metricas_no = calcular_metricas_nivel_no(top_n=10)

# 3. Visualização de centralidades
print("\n" + "="*80)
print("GERANDO VISUALIZAÇÕES DE CENTRALIDADES...")
print("="*80)
visualizar_centralidades(metricas=metricas_no)

# 4. Análise de comunidades
print("\n" + "="*80)
print("ANALISANDO COMUNIDADES...")
print("="*80)
resultado_comunidades = analisar_comunidades()

# 5. Visualização comparativa de comunidades
print("\n" + "="*80)
print("GERANDO COMPARAÇÃO VISUAL DE COMUNIDADES...")
print("="*80)
visualizar_comparacao_comunidades()

graus = [G.degree(n) for n in G.nodes()]
grau_medio = sum(graus) / len(graus)

print(f"Grau médio: {grau_medio:.2f}")
j = 0