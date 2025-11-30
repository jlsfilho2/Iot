# Prova de Conceito para Análise Preditiva de Hotspots Criminais e Otimização de Despacho de Viaturas
Autor: Fábio Viegas e João Luiz Scherer Filho
# Definição do Problema
Nos centros urbanos modernos, as operações de segurança pública são frequentemente reativas. Os centros de comando (como o COPOM) respondem a ocorrências à medida que são reportadas, muitas vezes com informações incompletas. O despacho de viaturas, um pilar dessa resposta, nem sempre é otimizado, baseando-se em zonas de patrulha fixas ou na percepção do operador sobre a unidade mais próxima, sem considerar a posição exata de todas as viaturas disponíveis em tempo real.
## Justificativa
A deficiência na análise de padrões criminais e a otimização subótima do despacho de recursos levam a um aumento no tempo de resposta. Cada segundo economizado na chegada a uma ocorrência crítica (como um disparo de arma de fogo) impacta diretamente a segurança dos cidadãos, a dos próprios policiais e a probabilidade de captura dos suspeitos
# Arquitetura e Metodologia
Como uma PoC acadêmica, o foco foi validar a lógica de IA sem o overhead de hardware físico ou pipelines de streaming complexos. O ambiente de desenvolvimento foi o Google Colab
# Mocka um dataset para Porto Alegre
Sensores de Disparo (ex: ShotSpotter): Dispositivos IoT que detectam o som de um disparo de arma de fogo e reportam a localização (latitude/longitude) e o horário.

GPS das Viaturas: Cada viatura reporta sua localização (latitude/longitude) e seu status (ex: 'disponível', 'em ocorrência') em tempo real.

Gera dados "Mockados" tanto dos sentores como das viaturas.

#Gera os dados os identificadores de disparos


```python
import pandas as pd
import numpy as np
import random
```


```python

print("Iniciando a geração dos datasets (Localização: Porto Alegre, RS)...")

# --- Funções de Geração ---

# Coordenadas do centro de Porto Alegre, RS
LAT_BASE = -30.066833
LON_BASE = -51.168570

CONFIGS = [{
    "zona" : "centro",
    "latitude" : -30.046078,
    "longitude" : -51.201298,
    "concentracao" : 0.05,
    "limite_guaiba" : -51.207239,
    "num_ocorrencias" : 1000,
    "viaturas" : 10
},{
    "zona" : "sul",
    "latitude" : -30.078930,
    "longitude" : -51.203834,
    "concentracao" : 0.04,
    "limite_guaiba" : -51.2418,
    "num_ocorrencias" : 750,
    "viaturas" : 10
},{
    "zona" : "extremo_sum",
    "latitude" : -30.173413,
    "longitude" : -51.150222,
    "concentracao" : 0.06,
    "limite_guaiba" : -51.241096,
    "num_ocorrencias" : 750,
    "viaturas" : 10
},{
    "zona" : "norte",
    "latitude" : -30.018886,
    "longitude" : -51.144584,
    "concentracao" : 0.04,
    "limite_guaiba" : -51.207239,
    "num_ocorrencias" : 1000,
    "viaturas" : 20
}]


def gerar_dados_shotspotter(num_ocorrencias=5000, num_hotspots=4, lat_base=LAT_BASE, lon_base=LON_BASE, ruido=0.6, limite_guaiba=-51.207239,concentracao=0.005):
    """
    Gera um DataFrame simulando dados de sensores de disparo (ShotSpotter).
    Cria "hotspots" concentrando ocorrências em áreas específicas.
    """
    np.random.seed(42)
    ocorrencias = []

    # Define os centros dos hotspots aleatoriamente ao redor da base
    hotspots_centers = []
    for _ in range(num_hotspots):
        # Desvio pequeno para os centros dos hotspots (ex: bairros próximos)
        hotspots_centers.append((lat_base + np.random.randn() * 0.03, lon_base + np.random.randn() * 0.03))

    for i in range(num_ocorrencias):
        # Decide se o ponto será em um hotspot ou "ruído"
        if random.random() > ruido:
            # Ponto dentro de um hotspot
            center = random.choice(hotspots_centers)
            lat = center[0] + np.random.randn() * concentracao # Desvio padrão pequeno (concentrado)
            lon = center[1] + np.random.randn() * concentracao
        else:
            # Ponto de "ruído" (aleatório pela cidade)
            # Desvio padrão maior para cobrir uma área mais ampla
            lat = lat_base + (np.random.rand() - 0.5) * concentracao * 3
            lon = lon_base + (np.random.rand() - 0.5) * concentracao * 3
        if lat < limite_guaiba:
            lat = limite_guaiba

        # Gera um timestamp aleatório nos últimos 30 dias
        timestamp = pd.Timestamp.now() - pd.to_timedelta(np.random.randint(0, 30*24*60), unit='m')

        ocorrencias.append({
            'id_ocorrencia': i,
            'timestamp': timestamp,
            'latitude': lat,
            'longitude': lon
        })

    
    return ocorrencias
```

    Iniciando a geração dos datasets (Localização: Porto Alegre, RS)...


# Gera dados das Viaturas


```python
def gerar_dados_viaturas(num_viaturas=50, lat_base=LAT_BASE, lon_base=LON_BASE,limite_guaiba=-51.207239):
    """
    Gera um DataFrame simulando a posição e o status das viaturas (GPS IoT).
    """
    np.random.seed(42)
    viaturas = []

    for i in range(num_viaturas):
        # Espalha as viaturas pela cidade
        lat = lat_base + (np.random.rand() - 0.5) * 0.1
        lon = lon_base + (np.random.rand() - 0.5) * 0.1
        if lat < limite_guaiba:
            lat = limite_guaiba

        # Define 70% como 'disponível' para o nosso teste
        if random.random() < 0.7:
            status = 'disponível'
        else:
            status = 'em_ocorrencia'

        viaturas.append({
            'id_viatura': f"VTR-{100 + i}",
            'latitude': lat,
            'longitude': lon,
            'status': status
        })

    return viaturas
```

#gera os datasets de ocorrência e viatura


```python


# Gerar os dados
# Ajustamos o número de hotspots e o ruído para uma boa visualização
df_viaturas = None
df_ocorrencias = None
for z in CONFIGS:
    ocorrencias = gerar_dados_shotspotter(num_ocorrencias=z.get("num_ocorrencias"), num_hotspots=5, lat_base=z.get("latitude"), lon_base=z.get("longitude"), ruido=0.5,limite_guaiba=z.get("limite_guaiba"))
    if df_ocorrencias is None:
        df_ocorrencias = pd.DataFrame(ocorrencias)
    else:
        df_ocorrencias = pd.concat([pd.DataFrame(ocorrencias, columns=df_ocorrencias.columns), df_ocorrencias], ignore_index=True)
    viaturas = gerar_dados_viaturas(num_viaturas=z.get("viaturas"), lat_base=z.get("latitude"), lon_base=z.get("longitude"),limite_guaiba=z.get("limite_guaiba"))
    if df_viaturas is None:
        df_viaturas = pd.DataFrame(viaturas)
    else:
        df_viaturas = pd.concat([pd.DataFrame(viaturas, columns=df_viaturas.columns), df_viaturas], ignore_index=True)
# Salvar em arquivos CSV
df_ocorrencias.to_csv('ocorrencias.csv', index=False)
df_viaturas.to_csv('viaturas.csv', index=False)

print("\nArquivos 'ocorrencias.csv' e 'viaturas.csv' (Porto Alegre) gerados com sucesso!")
print("Você pode vê-los no painel de arquivos à esquerda ou baixá-los.")

print("\n--- Amostra Ocorrências (ShotSpotter) ---")
print(df_ocorrencias.head())
print("\n--- Amostra Viaturas (GPS) ---")
print(df_viaturas.head())
```

# Instalação de bibliotecas (se necessário) e Imports


```python

!pip install folium
import pandas as pd
import numpy as np
import foliumdistancias_km
from folium.plugins import MarkerCluster
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
import random
from math import radians

print("Bibliotecas importadas com sucesso!")
```


#Carregamento dos Datasets (Lendo os arquivos CSV)


```python
print("Carregando datasets de 'ocorrencias.csv' e 'viaturas.csv'...")

try:
    df_ocorrencias = pd.read_csv('ocorrencias.csv')
    df_viaturas = pd.read_csv('viaturas.csv')
except FileNotFoundError:
    print("\nERRO: Arquivos CSV não encontrados.")
    print("Por favor, execute o 'Bloco 1: Gerador de Datasets (Porto Alegre)' primeiro.")
    # Interrompe a execução se os arquivos não existirem
    raise

# IMPORTANTE: Converter a coluna de timestamp de volta para datetime
df_ocorrencias['timestamp'] = pd.to_datetime(df_ocorrencias['timestamp'])


# --------------------

print("\nDatasets carregados com sucesso!")
print(f"{len(df_ocorrencias)} ocorrências e {len(df_viaturas)} viaturas lidas.")

print("\n--- Amostra Ocorrências (do CSV) ---")
print(df_ocorrencias.head())
```

#Análise de Hotspots (IA - Parte 1: Clustering)


```python

```


```python
print("\nIniciando análise de IA (DBSCAN) para encontrar hotspots...")


# 1. Preparar os dados para o DBSCAN
coords_ocorrencias = df_ocorrencias[['latitude', 'longitude']].apply(np.radians).values

# 2. Configurar o DBSCAN
kms_por_radiano = 6371.0087714
epsilon = 0.5 / kms_por_radiano # Raio de 500 metros
db = DBSCAN(eps=epsilon, min_samples=20, algorithm='ball_tree', metric='haversine').fit(coords_ocorrencias)

# 3. Adicionar os resultados ao DataFrame
df_ocorrencias['hotspot_id'] = db.labels_
num_hotspots = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

print(f"Análise concluída. Encontrados {num_hotspots} hotspots principais.")
print(df_ocorrencias['hotspot_id'].value_counts())
```

# Visualização (Dashboard Interativo com Folium)


```python


print("\nGerando mapa interativo (Porto Alegre)...")

# O Folium agora usará as coordenadas de POA definidas na Célula 2
mapa = folium.Map(location=[LAT_BASE, LON_BASE], zoom_start=12)

# 1. Adicionar os Hotspots ao Mapa
hotspots_clusters = df_ocorrencias[df_ocorrencias['hotspot_id'] != -1]
marker_cluster = MarkerCluster(name="Ocorrências (Hotspots)").add_to(mapa)

cores = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue']
cores_mapa = {cluster_id: cores[i % len(cores)] for i, cluster_id in enumerate(hotspots_clusters['hotspot_id'].unique())}

for idx, row in hotspots_clusters.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color=cores_mapa[row['hotspot_id']],
        fill=True,
        fill_color=cores_mapa[row['hotspot_id']],
        fill_opacity=0.7,
        popup=f"ID Ocorrência: {row['id_ocorrencia']}<br>Hotspot: {row['hotspot_id']}"
    ).add_to(marker_cluster)

# 2. Adicionar as Viaturas ao Mapa
viaturas_group = folium.FeatureGroup(name="Viaturas").add_to(mapa)

for idx, row in df_viaturas.iterrows():
    if row['status'] == 'disponível':
        icon_color = 'blue'
        icon_type = 'glyphicon-ok-sign'
    else:
        icon_color = 'red'
        icon_type = 'glyphicon-remove-sign'

    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"<b>{row['id_viatura']}</b><br>Status: {row['status']}",
        icon=folium.Icon(color=icon_color, icon=icon_type, prefix='glyphicon')
    ).add_to(viaturas_group)

folium.LayerControl().add_to(mapa)

print("Mapa gerado! (Role para baixo para ver)")
display(mapa)
```


#Simulação de Ação Inteligente (IA - Parte 2: Despacho)


```python

def despachar_viatura_mais_proxima(nova_ocorrencia, df_viaturas_atual):
    viaturas_disponiveis = df_viaturas_atual[df_viaturas_atual['status'] == 'disponível'].copy()

    if len(viaturas_disponiveis) == 0:
        return None, None, None

    ocorrencia_rad = [radians(nova_ocorrencia['latitude']), radians(nova_ocorrencia['longitude'])]
    viaturas_rad = viaturas_disponiveis[['latitude', 'longitude']].apply(np.radians).values

    distancias = haversine_distances([ocorrencia_rad], viaturas_rad)
    distancias_km = distancias[0] * 6371.0088

    idx_mais_proxima = np.argmin(distancias_km)
    distancia_minima = distancias_km[idx_mais_proxima]

    viatura_selecionada = viaturas_disponiveis.iloc[idx_mais_proxima]

    return viatura_selecionada, distancia_minima
```

# Executando a Simulação de Despacho


```python


print("\n--- SIMULAÇÃO DE DESPACHO INTELIGENTE (PORTO ALEGRE) ---")

# 1. Simular um novo evento IoT (disparo)
# --- MUDANÇA AQUI ---
# (Vamos pegar uma localização aleatória perto da área base de Porto Alegre)
nova_ocorrencia = {
    'latitude': LAT_BASE + (np.random.rand() - 0.5) * 0.1,
    'longitude': LON_BASE + (np.random.rand() - 0.5) * 0.1
}
# --------------------

print(f"Novo disparo detectado em: Lat {nova_ocorrencia['latitude']:.4f}, Lon {nova_ocorrencia['longitude']:.4f}")

# 2. Rodar a IA para tomar a decisão (usando os dados de POA)
viatura, distancia = despachar_viatura_mais_proxima(nova_ocorrencia, df_viaturas)

if viatura is not None:
    print(f"\n--- AÇÃO INTELIGENTE ---")
    print(f"Despachando a viatura {viatura['id_viatura']}!")
    print(f"Distância estimada: {distancia:.2f} km.")

    # 3. Atualizar o mapa com a nova ocorrência e a rota
    mapa_despacho = mapa

    folium.Marker(
        location=[nova_ocorrencia['latitude'], nova_ocorrencia['longitude']],
        popup="<b>NOVA OCORRÊNCIA</b>",
        icon=folium.Icon(color='red', icon='exclamation-sign', prefix='glyphicon')
    ).add_to(mapa_despacho)

    folium.PolyLine(
        locations=[
            (viatura['latitude'], viatura['longitude']),
            (nova_ocorrencia['latitude'], nova_ocorrencia['longitude'])
        ],
        color='red',
        weight=3,
        opacity=0.8,
        dash_array='5, 10'
    ).add_to(mapa_despacho)

    print("\nMapa de despacho gerado! (Role para baixo para ver)")
    display(mapa_despacho)

else:
    print("NENHUMA VIATURA DISPONÍVEL NO MOMENTO.")
```
