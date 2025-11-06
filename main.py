import requests
import json
import networkx as nx
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import csv
import os
import matplotlib
import scipy

G = nx.Graph()
votacoes_por_deputado = {}
votacoes_ids = []
deputadosFaltantes = [] #deputados que não vieram na requisição de deputados, mas apareceram como votantes em alguma votação
deputadosJaVistos = [] #lista controle pra agilizar a checagem

qtdVertices = 0

def salvar_deputado_no_csv(atributo_deputado, caminho_csv='deputados_56_legislatura_atributos.csv'):
    """
    Salva ou adiciona um deputado ao arquivo CSV.
    Se o arquivo não existir, cria com cabeçalho. Se existir, adiciona a linha.
    """
    arquivo_existe = os.path.exists(caminho_csv) and os.path.getsize(caminho_csv) > 0
    
    fieldnames = list(atributo_deputado.keys())
    
    with open(caminho_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not arquivo_existe:
            writer.writeheader()
        writer.writerow(atributo_deputado)

def criar_Vertices():
    """Faz as requisita os deputados da 56ª legislatura, salva eles e seus dados associados em um .csv e adiciona vértices no grafo."""
    # a legislatura atual é a 57, mas pela completude dos dados, vou usar a 56
    x = requests.get('https://dadosabertos.camara.leg.br/api/v2/deputados?idLegislatura=56')
    deputados = x.json()

    # Limpar arquivo CSV existente antes de adicionar todos os deputados
    caminho_csv = 'deputados_56_legislatura_atributos.csv'
    if os.path.exists(caminho_csv):
        os.remove(caminho_csv)

    qtd_vertices = 0
    for deputado in deputados['dados']:
        atributo_deputado = deputado.copy()
        atributo_deputado.pop('uri')
        atributo_deputado.pop('uriPartido')
        atributo_deputado.pop('urlFoto')
        atributo_deputado.pop('email')

        idDeputado = atributo_deputado['id']
        votacoes_por_deputado[idDeputado] = []
        
        # Salvar deputado no CSV usando a função reutilizável
        salvar_deputado_no_csv(atributo_deputado, caminho_csv)
        qtd_vertices = qtd_vertices + 1


def carregar_vertices_de_csv(caminho_csv):
    """
    Lê um arquivo CSV com atributos dos deputados e popula o grafo G,
    além de atualizar nomesDeputados e votacoes_por_deputado.
    """
    # limpar estruturas atuais
    G.clear()
    votacoes_por_deputado.clear()

    with open(caminho_csv, newline='', encoding='utf-8') as csvfile:
        global qtdVertices
        reader = csv.DictReader(csvfile)
        idx_no = 0
        for row in reader:
            # tentar converter id para inteiro, se existir
            idDeputado = row.get('id')
            try:
                idDeputado_int = int(idDeputado) if idDeputado is not None else None
            except ValueError:
                idDeputado_int = None

            if idDeputado_int is not None:
                votacoes_por_deputado[idDeputado_int] = []

            # adicionar nó com atributos do CSV
            G.add_node(idx_no, **row)
            qtdVertices+=1
            idx_no += 1


def obterVotacoes(dataInicio, dataFim):
    """Requisita os dados das votações, e guarda o voto de cada deputado."""
    x = requests.get(f'https://dadosabertos.camara.leg.br/api/v2/votacoes?dataFim={dataFim}&dataInicio={dataInicio}')
    votacoes = x.json()
    
    # Salvar dados das votações em CSV
    if votacoes.get('dados'):
        arquivo_votacoes = 'votacoes.csv'
        arquivo_existe = False
        try:
            with open(arquivo_votacoes, 'r', encoding='utf-8'):
                arquivo_existe = True
        except FileNotFoundError:
            arquivo_existe = False
        
        with open(arquivo_votacoes, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(votacoes['dados'][0].keys()) if votacoes['dados'] else []
            if fieldnames:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not arquivo_existe:
                    writer.writeheader()
                writer.writerows(votacoes['dados'])
        
        # Para cada votação, buscar os votos e salvar
        for votacao in votacoes['dados']:
            id_votacao = votacao['id']
            try:
                x_votos = requests.get(f"https://dadosabertos.camara.leg.br/api/v2/votacoes/{id_votacao}/votos")
            except:
                with open(arquivo_votacoes, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer()
                    writer.writerow(id_votacao)
                continue
            votos = x_votos.json()
            
            # Salvar dados dos votos em CSV
            if votos.get('dados'):
                for voto in votos['dados']:
                        idDeputado = voto['deputado_']['id']
                        resultadoVoto = voto['tipoVoto']
                        if idDeputado not in votacoes_por_deputado:
                            votacoes_por_deputado[idDeputado] = []
                            if idDeputado not in deputadosJaVistos:
                                deputadosJaVistos.append(idDeputado)
                                deputadosFaltantes.append(idDeputado)
                        
                        votacoesDeputadoCopy = list(votacoes_por_deputado[idDeputado])
                        votacoesDeputadoCopy.append({id_votacao: resultadoVoto})
                        votacoes_por_deputado[idDeputado] = votacoesDeputadoCopy
def buscarTemaVotacao(dataInicio, dataFim):
    x = requests.get(f'https://dadosabertos.camara.leg.br/api/v2/votacoes?dataFim={dataFim}&dataInicio={dataInicio}')
    votacoes = x.json()
    if votacoes.get('dados'):
        for votacao in votacoes['dados']:
            #id_proposicao = votacao['idProposicao']
            #x = requests.get(f'https://dadosabertos.camara.leg.br/api/v2/proposicoes/{id_proposicao}/temas')

            if votacao['uriProposicaoObjeto'] != None or votacao['proposicaoObjeto'] != None:
                j = 0
                #TODO: nem toda votação tem uma proposição, e só proposição tem tema
            i = 0

def adicionarTemaVotacoes():
    cont = 1
    inicio_total = date(2019, 2, 1)
    fim_total = date(2023, 1, 31)
    dInicio = inicio_total
    while dInicio <= fim_total:
        # Fim do intervalo: 3 meses depois, menos 1 dia
        dFim = dInicio + relativedelta(months=3) - relativedelta(days=1)
        
        # Se ultrapassar o fim total, ajustar
        if dFim > fim_total:
            dFim = fim_total
        
        if cont == 1:
            cont+=1
            continue
        
        buscarTemaVotacao(dInicio, dFim)
        
        # Próximo intervalo começa no dia seguinte ao final do atual
        dInicio = dFim + relativedelta(days=1)

def obterVotacoesDaLegislatura():
    """
    Chama o método obterVotacoes para todos os períodos de 3 meses entre o início e o fim da legislatura.
    """
    inicio_total = date(2019, 2, 1)
    fim_total = date(2023, 1, 31)
    dInicio = inicio_total
    while dInicio <= fim_total:
        # Fim do intervalo: 3 meses depois, menos 1 dia
        dFim = dInicio + relativedelta(months=3) - relativedelta(days=1)
        
        # Se ultrapassar o fim total, ajustar
        if dFim > fim_total:
            dFim = fim_total
        
        obterVotacoes(dInicio, dFim)
        
        # Próximo intervalo começa no dia seguinte ao final do atual
        dInicio = dFim + relativedelta(days=1)

def salvarVotacoes(caminho_csv='votacoes_por_deputado.csv'):
    """
    Salva o dicionário votacoes_por_deputado em um arquivo CSV.
    Formato do CSV: id_deputado, id_votacao, tipo_voto
    """
    with open(caminho_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id_deputado', 'id_votacao', 'tipo_voto']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for id_deputado, votacoes in votacoes_por_deputado.items():
            for votacao in votacoes:
                # Cada votacao é um dict no formato {id_votacao: tipo_voto}
                for id_votacao, tipo_voto in votacao.items():
                    writer.writerow({
                        'id_deputado': id_deputado,
                        'id_votacao': id_votacao,
                        'tipo_voto': tipo_voto
                    })

def carregarVotacoesDeCSV(caminho_csv='votacoes_por_deputado.csv'):
    """
    Lê um arquivo CSV e repopula o dicionário votacoes_por_deputado.
    Formato esperado do CSV: id_deputado, id_votacao, tipo_voto
    """
    votacoes_por_deputado.clear()
    
    try:
        with open(caminho_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                id_deputado = int(row['id_deputado'])
                id_votacao = str(row['id_votacao'])
                tipo_voto = row['tipo_voto']
                
                # Se o deputado ainda não existe no dict, criar lista vazia
                if id_deputado not in votacoes_por_deputado:
                    votacoes_por_deputado[id_deputado] = []
                
                # Adicionar a votação no formato {id_votacao: tipo_voto}
                votacoes_por_deputado[id_deputado].append({id_votacao: tipo_voto})
    except FileNotFoundError:
        print(f"Arquivo {caminho_csv} não encontrado. O dicionário votacoes_por_deputado permanece vazio.")

def obterVertice(identificador):
    global qtdVertices
    vertices = G.nodes(data= True)
    for vertice in vertices:
        if vertice[1]['id'] == str(identificador):
            return vertice[0]
    url = f'https://dadosabertos.camara.leg.br/api/v2/deputados/{identificador}'
    x = requests.get(url)
    deputado = x.json()

    dados = deputado['dados']['ultimoStatus']

    # Formatar atributo_deputado para salvar no CSV (igual ao formato usado em criarVertice)
    atributo_deputado = {
        'id': dados['id'],
        'nome': dados['nome'],
        'siglaPartido': dados['siglaPartido'],
        'siglaUf': dados['siglaUf'],
        'idLegislatura': dados.get('idLegislatura', 56)  # Usar 56 como padrão se não estiver disponível
    }

    # Salvar deputado no CSV usando a função reutilizável
    salvar_deputado_no_csv(atributo_deputado)

    G.add_node(qtdVertices)
    G.nodes[qtdVertices]['id'] = dados['id']
    G.nodes[qtdVertices]['nome'] = dados['nome']
    G.nodes[qtdVertices]['siglaPartido'] = dados['siglaPartido']
    G.nodes[qtdVertices]['siglaUf'] = dados['siglaUf']
    qtdVertices+=1
    return qtdVertices-1

def calcularPesosArestas():
    """
    Calcula a probabilidade de dois deputados votarem juntos em uma votação e cria uma aresta com peso entre os dois.
    """

    for idDeputadoA in votacoes_por_deputado:
        numVerticeA = obterVertice(idDeputadoA)
        for idDeputadoB in votacoes_por_deputado:
            numVerticeB = obterVertice(idDeputadoB)
            probabilidade = 0
            totalVotacoes = 0 #votacoes que os dois participaram
            votosIguais = 0
            if idDeputadoA != idDeputadoB:
                votosA = votacoes_por_deputado[idDeputadoA]
                votosB = votacoes_por_deputado[idDeputadoB]

                totalVotacoes = 0 #votacoes que os dois participaram
                votosIguais = 0 #votacoes em que os dois votaram igual

                for votoA in votosA:
                    idVotacaoA = list(dict(votoA).keys())[0]
                    for votoB in votosB:
                        idVotacaoB = list(dict(votoB).keys())[0]
                        if idVotacaoA == idVotacaoB:
                            totalVotacoes += 1
                            tipoVotoA = dict(votoA).get(idVotacaoA)
                            tipoVotoB = dict(votoB).get(idVotacaoB)
                            if tipoVotoA == tipoVotoB:
                                votosIguais += 1
            probabilidade = 0 if totalVotacoes == 0 else votosIguais/totalVotacoes
            if probabilidade == 1:
                G.add_edge(numVerticeA, numVerticeB, weight=probabilidade )
                #G.add_edge(numVerticeA, numVerticeB)
                #nx.set_edge_attributes(G, {(numVerticeA, numVerticeB): {"afinidade": probabilidade}})            k = 0

#criar_Vertices()
#salvarVotacoes()

""" carregar_vertices_de_csv("deputados_56_legislatura_atributos.csv")
carregarVotacoesDeCSV()
calcularPesosArestas()

nx.write_weighted_edgelist(G, "grafoDeputados100.weighted.edgelist") """

#adicionarTemaVotacoes()

G = nx.read_weighted_edgelist("grafoDeputados100.weighted.edgelist")
nx.draw(G, pos=nx.spring_layout(G), with_labels=True)

""" a rede tá com só uma componente conexa

ideias: 
    fazer a rede só com os deputados que tinham no início da legislatua
    fazer a rede só com os deputados que votaram nas mesmas propostas

    TODO: consertei(?) o jeito como tava calculando o peso das arestas, e deu 38 componentes conexas.
    quando rodei de novo lendo do arquivo, a rede tava com só 1 de novo. tem algum problema com como tá salvando a rede.
"""
i = 0