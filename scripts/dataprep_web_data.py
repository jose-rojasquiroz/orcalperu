import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import json
from pathlib import Path
from shapely.geometry import box, LineString
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

INPUT_GPKG = 'data/pu-principalesciudades-WGS84.gpkg'
OUTPUT_DIR = Path('docs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Crear subdirectorios
(OUTPUT_DIR / 'fichas').mkdir(exist_ok=True)
(OUTPUT_DIR / 'data').mkdir(exist_ok=True)
(OUTPUT_DIR / 'graficos').mkdir(exist_ok=True)
(OUTPUT_DIR / 'mapas').mkdir(exist_ok=True)

# Colores por región
COLORES = {
    'Costa': '#c7cb52',
    'Sierra': '#98692e',
    'Selva': '#62a162'
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calcular_densidad(gdf):
    """Calcula el área en km² y la densidad poblacional"""
    # Convertir a CRS métrico (UTM zone 18S para Perú)
    gdf_metric = gdf.to_crs('EPSG:32718')
    gdf['area_km2'] = gdf_metric.geometry.area / 1_000_000
    gdf['densidad'] = gdf['POB17'] / gdf['area_km2']
    return gdf

def get_bearings_from_polygon(city_polygon, city_name):
    """Obtiene los bearings de las calles de una ciudad (soporta MultiPolygon)"""
    from shapely.geometry import MultiPolygon
    import networkx as nx
    try:
        print(f"  Descargando red de calles para {city_name}...")
        
        # Si es MultiPolygon, descargar para cada polígono y combinar
        if isinstance(city_polygon, MultiPolygon):
            graphs = []
            for i, poly in enumerate(city_polygon.geoms):
                try:
                    g = ox.graph_from_polygon(poly, network_type='drive')
                    if g and len(g.nodes) > 0:
                        graphs.append(g)
                except Exception as e_poly:
                    print(f"    Polígono {i+1}: sin calles o error")
                    continue
            
            if not graphs:
                print(f"  ✗ {city_name}: No se pudo descargar ninguna red")
                return None, None
            
            # Combinar grafos usando networkx compose
            graph = graphs[0]
            for g in graphs[1:]:
                graph = nx.compose(graph, g)
        else:
            graph = ox.graph_from_polygon(city_polygon, network_type='drive')
        
        graph = ox.bearing.add_edge_bearings(graph)
        bearings = [d['bearing'] for u, v, k, d in graph.edges(keys=True, data=True) if 'bearing' in d]
        print(f"  ✓ {city_name}: {len(bearings)} segmentos de calles")
        return bearings, graph
    except Exception as e:
        print(f"  ✗ Error en {city_name}: {e}")
        return None, None

def create_street_map(graph, city_name, region, output_path, map_extent=None):
    """Crea y guarda el mapa de la red de calles"""
    try:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        
        # Plot del grafo
        ox.plot_graph(graph, ax=ax, node_size=0, edge_color='#333333', 
                      edge_linewidth=0.5, bgcolor='white', show=False, close=False)
        if map_extent:
            zoomed = zoom_extent(map_extent, factor=1/1.6)
            xmin, xmax, ymin, ymax = zoomed
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            add_scale_bar(ax, zoomed, graph.graph.get('crs', 'EPSG:4326') if hasattr(graph, 'graph') else 'EPSG:4326')

        # Remover ejes
        ax.set_axis_off()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
        plt.close()
        
        print(f"  ✓ Mapa guardado: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error creando mapa para {city_name}: {e}")
        return False

def create_polar_plot(bearings, city_name, region, output_path):
    """Crea y guarda el gráfico polar de orientación"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    color = COLORES[region]
    direcciones = ['N', '', 'E', '', 'S', '', 'O', '']
    
    # Histograma
    ax.hist(np.deg2rad(bearings), bins=36, color=color, alpha=0.75, edgecolor='white', linewidth=1)
    
    # Configuración del gráfico
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 8, endpoint=False))
    ax.set_xticklabels(direcciones, fontsize=14, fontweight='bold')
    
    # Fondo transparente
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Gráfico guardado: {output_path.name}")

def create_ficha(city_data, bearings, graph, output_path, map_extent=None):
    """Crea una ficha completa con dimensiones precisas en cm."""
    # Dimensiones en cm (convertidas a pulgadas)
    width_cm = 24.16
    height_cm = 19.05
    dpi = 600
    fig = plt.figure(figsize=(width_cm/2.54, height_cm/2.54), dpi=dpi)
    fig.patch.set_facecolor('white')
    
    # Márgenes y áreas (en fracción de figura)
    title_h = 2.05 / height_cm
    map_h = 15.64 / height_cm
    map_w = 14.64 / width_cm  # Reducido 1 cm desde 15.64
    polar_h = 8.56 / height_cm
    polar_w = 8.52 / width_cm
    info_h = 2.05 / height_cm
    gap_below_polar = 0.98 / height_cm
    gap_top_polar = 1.96 / height_cm  # Bajada adicional del gráfico polar
    
    # Alineación derecha para el gráfico polar e info (pegado al borde derecho)
    polar_x = 1 - polar_w
    
    base_crs = graph.graph.get('crs', 'EPSG:4326') if hasattr(graph, 'graph') else 'EPSG:4326'
    
    # TÍTULO (centrado en toda la anchura)
    ax_title = fig.add_axes([0, 1 - title_h, 1, title_h])
    ax_title.axis('off')
    city_name = city_data['CIUDAD']
    city_name_formatted = ' '.join(word.capitalize() for word in city_name.split())
    ax_title.text(0.5, 0.5, city_name_formatted, fontsize=32, fontweight='bold',
                  ha='center', va='center', transform=ax_title.transAxes)
    
    # MAPA DE CALLES (izquierda)
    ax_mapa = fig.add_axes([0, 1 - title_h - map_h, map_w, map_h])
    # Reducir ancho de líneas para Lima Metropolitana
    is_lima = city_data['CIUDAD'].upper() == 'LIMA METROPOLITANA'
    linewidth = 0.5 / 3 if is_lima else 0.5
    try:
        ox.plot_graph(graph, ax=ax_mapa, node_size=0, edge_color='#333333',
                      edge_linewidth=linewidth, bgcolor='white', show=False, close=False)
        if map_extent:
            zoomed = zoom_extent(map_extent, factor=1/1.6)
            xmin, xmax, ymin, ymax = zoomed
            ax_mapa.set_xlim(xmin, xmax)
            ax_mapa.set_ylim(ymin, ymax)
            add_scale_bar(ax_mapa, zoomed, base_crs)
        ax_mapa.set_axis_off()
    except:
        ax_mapa.text(0.5, 0.5, 'Mapa no disponible', ha='center', va='center')
        ax_mapa.set_axis_off()
    
    # GRÁFICO POLAR (derecha, arriba, con bajada de 1.96 cm)
    ax_polar = fig.add_axes([polar_x, 1 - title_h - gap_top_polar - polar_h, polar_w, polar_h],
                            projection='polar')
    color = COLORES[city_data['REGNAT']]
    direcciones = ['N', '', 'E', '', 'S', '', 'O', '']
    ax_polar.hist(np.deg2rad(bearings), bins=36, color=color, alpha=0.75,
                  edgecolor='white', linewidth=1)
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_yticklabels([])
    ax_polar.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 8, endpoint=False))
    ax_polar.set_xticklabels(direcciones, fontsize=10, fontweight='bold')
    ax_polar.patch.set_facecolor('white')
    
    # INFORMACIÓN (derecha, abajo del gráfico polar)
    ax_info = fig.add_axes([polar_x, 1 - title_h - gap_top_polar - polar_h - gap_below_polar - info_h,
                            polar_w, info_h])
    ax_info.axis('off')
    poblacion_int = int(city_data['POB17'])
    info_text = f"""Población: {poblacion_int:,} habitantes
Área urbana: {city_data['area_km2']:.0f} km²
Densidad: {city_data['densidad']:.0f} hab/km²"""
    ax_info.text(0.5, 0.5, info_text, fontsize=11, ha='center', va='center',
                 transform=ax_info.transAxes, family='monospace')
    
    # FOOTER
    fig.text(0.5, 0.01, 'Elaborado por José Rojas-Quiroz | Datos: INEI 2017, OpenStreetMap y OSMnx',
             ha='center', fontsize=9, color='gray')
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Ficha guardada: {output_path.name}")

def process_city(idx, city_row, gdf_cities, map_extent, is_top3=False):
    """Procesa una ciudad individual. is_top3 define si se guardan mapa/gráfico individuales."""
    city_name = city_row['CIUDAD']
    city_polygon = city_row['geometry']
    region = city_row['REGNAT']
    
    print(f"\n[{idx+1}/{len(gdf_cities)}] Procesando: {city_name} ({region})")
    
    # Obtener bearings y grafo
    bearings, graph = get_bearings_from_polygon(city_polygon, city_name)
    
    if bearings is None or len(bearings) == 0:
        print(f"  ✗ Sin datos para {city_name}")
        return None
    
    mapa_path = None
    polar_path = None

    # Solo guardar mapas/gráficos individuales para el top 3 por región (sin Lima Metropolitana)
    if is_top3:
        mapa_path = OUTPUT_DIR / 'mapas' / f'{city_name.replace(" ", "_")}_mapa.png'
        polar_path = OUTPUT_DIR / 'graficos' / f'{city_name.replace(" ", "_")}_polar.png'
        create_street_map(graph, city_name, region, mapa_path, map_extent)
        create_polar_plot(bearings, city_name, region, polar_path)
    
    # Crear ficha completa (ahora incluye el mapa)
    ficha_path = OUTPUT_DIR / 'fichas' / f'{city_name.replace(" ", "_")}_ficha.png'
    create_ficha(city_row, bearings, graph, ficha_path, map_extent)
    
    # Preparar datos para JSON
    city_data = {
        'nombre': city_name,
        'region': region,
        'departamento': city_row['DEPARTAMENTO'],
        'provincia': city_row['PROVINCIA'],
        'poblacion': int(city_row['POB17']),
        'area_km2': float(city_row['area_km2']),
        'densidad': float(city_row['densidad']),
        'bearings': [float(b) for b in bearings],
        'num_segmentos': len(bearings),
        'mapa_calles': f'mapas/{city_name.replace(" ", "_")}_mapa.png' if mapa_path else None,
        'grafico_polar': f'graficos/{city_name.replace(" ", "_")}_polar.png' if polar_path else None,
        'ficha': f'fichas/{city_name.replace(" ", "_")}_ficha.png'
    }
    
    return city_data, graph


def select_top_cities(gdf):
    """Selecciona top 3 ciudades por población por región (Costa, Sierra, Selva) excluyendo Lima Metropolitana."""
    gdf_no_lima = gdf[gdf['CIUDAD'].str.upper() != 'LIMA METROPOLITANA']
    top_cities = []
    for region in ['Costa', 'Sierra', 'Selva']:
        region_cities = gdf_no_lima[gdf_no_lima['REGNAT'] == region]
        top_cities.append(region_cities.nlargest(3, 'POB17'))
    return pd.concat(top_cities)


def assign_scale_group(gdf):
    """Divide en 5 grupos de escala: grupo 5 exclusivo para Lima Metropolitana; grupos 1-4 por cuantiles de área sin Lima."""
    gdf = gdf.copy()
    mask_lima = gdf['CIUDAD'].str.upper() == 'LIMA METROPOLITANA'

    gdf_no_lima = gdf[~mask_lima]
    q25, q50, q75 = gdf_no_lima['area_km2'].quantile([0.25, 0.50, 0.75])

    def group_row(row):
        if row['CIUDAD'].upper() == 'LIMA METROPOLITANA':
            return 5
        area = row['area_km2']
        if area >= q75:
            return 1
        if area >= q50:
            return 2
        if area >= q25:
            return 3
        return 4

    gdf['grupo_escala'] = gdf.apply(group_row, axis=1)
    return gdf


def compute_group_half_extent_m(gdf):
    """Calcula la mitad del lado del cuadro (en metros) por grupo de escala."""
    half_extents = {}
    gdf_metric = gdf.to_crs('EPSG:32718')
    for grupo in sorted(gdf['grupo_escala'].unique()):
        subset = gdf_metric[gdf_metric['grupo_escala'] == grupo]
        if subset.empty:
            continue
        bounds = subset.bounds
        max_dim = np.maximum(bounds['maxx'] - bounds['minx'], bounds['maxy'] - bounds['miny'])
        half_extents[grupo] = max_dim.max() / 2
    return half_extents


def build_map_extent(city_polygon, half_size_m, base_crs):
    """Genera una caja cuadrada centrada en la ciudad con el tamaño común, en coordenadas originales."""
    geom_metric = gpd.GeoSeries([city_polygon], crs=base_crs).to_crs('EPSG:32718').iloc[0]
    cx, cy = geom_metric.centroid.x, geom_metric.centroid.y
    bbox_metric = box(cx - half_size_m, cy - half_size_m, cx + half_size_m, cy + half_size_m)
    bbox_wgs = gpd.GeoSeries([bbox_metric], crs='EPSG:32718').to_crs(base_crs)
    minx, miny, maxx, maxy = bbox_wgs.total_bounds
    return (minx, maxx, miny, maxy)


def zoom_extent(extent, factor=1.0):
    """Devuelve un extent escalado alrededor del centro (factor <1 acerca, >1 aleja)."""
    xmin, xmax, ymin, ymax = extent
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    half_x = (xmax - xmin) / 2 * factor
    half_y = (ymax - ymin) / 2 * factor
    return (cx - half_x, cx + half_x, cy - half_y, cy + half_y)


def choose_scale_length(width_m):
    """Elige una longitud de barra de escala agradable basada en el ancho disponible."""
    nice_km = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    target = width_m / 4
    candidates = [k * 1000 for k in nice_km if k * 1000 <= target]
    if candidates:
        return max(candidates)
    return min(nice_km) * 1000


def add_scale_bar(ax, map_extent, base_crs):
    """Dibuja una escala gráfica simple en la esquina inferior izquierda del mapa."""
    if base_crs is None:
        base_crs = 'EPSG:4326'
    xmin, xmax, ymin, ymax = map_extent
    bbox_wgs = box(xmin, ymin, xmax, ymax)
    bbox_metric = gpd.GeoSeries([bbox_wgs], crs=base_crs).to_crs('EPSG:32718').iloc[0]
    minx_m, miny_m, maxx_m, maxy_m = bbox_metric.bounds
    width_m = maxx_m - minx_m
    height_m = maxy_m - miny_m
    bar_len_m = choose_scale_length(width_m)

    x_start = minx_m + width_m * 0.05
    y_pos = miny_m + height_m * 0.08
    x_end = x_start + bar_len_m

    line_metric = LineString([(x_start, y_pos), (x_end, y_pos)])
    line_wgs = gpd.GeoSeries([line_metric], crs='EPSG:32718').to_crs(base_crs).iloc[0]
    xs, ys = line_wgs.xy

    height_deg = ymax - ymin
    tick = height_deg * 0.01
    text_offset = height_deg * 0.02

    ax.plot(xs, ys, color='black', linewidth=2)
    ax.plot([xs[0], xs[0]], [ys[0] - tick, ys[0] + tick], color='black', linewidth=1)
    ax.plot([xs[1], xs[1]], [ys[1] - tick, ys[1] + tick], color='black', linewidth=1)

    label_km = bar_len_m / 1000
    label = f"{label_km:g} km"
    ax.text((xs[0] + xs[1]) / 2, ys[0] + text_offset, label,
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    print("="*70)
    print("PRE-PROCESAMIENTO DE DATOS - CIUDADES DEL PERÚ")
    print("="*70)
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos del GeoPackage...")
    gdf_cities = gpd.read_file(INPUT_GPKG)
    gdf_cities['geometry'] = gdf_cities['geometry'].buffer(0)
    gdf_cities = gdf_cities[gdf_cities.geometry.is_valid]
    print(f"  ✓ {len(gdf_cities)} ciudades cargadas")
    
    # 2. Calcular densidades
    print("\n[2/5] Calculando áreas y densidades poblacionales...")
    gdf_cities = calcular_densidad(gdf_cities)
    print(f"  ✓ Densidades calculadas")

    # 2b. Seleccionar top 3 por región (excluyendo Lima Metropolitana) para mapas/gráficos individuales
    gdf_top3 = select_top_cities(gdf_cities)
    top3_names = set(gdf_top3['CIUDAD'])
    print(f"  ✓ Seleccionadas {len(gdf_top3)} ciudades (top 3 por región, sin Lima Metropolitana)")

    # 2c. Asignar grupos de escala por población (4 grupos, grupo 1 = ciudades más grandes)
    gdf_cities = assign_scale_group(gdf_cities)
    half_extents = compute_group_half_extent_m(gdf_cities)

    # 2d. Calcular extensión por ciudad según su grupo
    map_extents = {}
    for _, row in gdf_cities.iterrows():
        grupo = row['grupo_escala']
        half = half_extents.get(grupo)
        if half:
            map_extents[row['CIUDAD']] = build_map_extent(row['geometry'], half, gdf_cities.crs)
    print(f"  ✓ Extensiones calculadas por grupos de escala")
    
    # 3. Procesar ciudades
    print("\n[3/5] Procesando ciudades (descargando redes y generando gráficos/mapas)...")
    print("  Nota: Este proceso puede tomar varios minutos...\n")
    
    all_city_data = []
    all_graphs = []
    
    # Procesamiento secuencial (más estable que paralelo para OSMnx)
    for idx, (_, city_row) in enumerate(gdf_cities.iterrows()):
        city_extent = map_extents.get(city_row['CIUDAD'])
        is_top3 = city_row['CIUDAD'] in top3_names
        result = process_city(idx, city_row, gdf_cities, city_extent, is_top3=is_top3)
        if result:
            city_data, graph = result
            all_city_data.append(city_data)
            all_graphs.append((city_data['nombre'], graph))
    
    print(f"\n  ✓ {len(all_city_data)} ciudades procesadas exitosamente")
    
    # 4. Guardar datos JSON
    print("\n[4/5] Guardando datos JSON...")
    
    # JSON general con todas las ciudades
    with open(OUTPUT_DIR / 'data' / 'ciudades.json', 'w', encoding='utf-8') as f:
        json.dump(all_city_data, f, ensure_ascii=False, indent=2)
    print("  ✓ ciudades.json")
    
    # JSON por región (top 3)
    for region in ['Costa', 'Sierra', 'Selva']:
        region_data = [c for c in all_city_data if c['region'] == region]
        region_data.sort(key=lambda x: x['poblacion'], reverse=True)
        top_3 = region_data[:3]
        
        with open(OUTPUT_DIR / 'data' / f'{region.lower()}_top3.json', 'w', encoding='utf-8') as f:
            json.dump(top_3, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {region.lower()}_top3.json")
    
    # 5. Exportar GeoPackages
    print("\n[5/5] Exportando GeoPackage unificado...")
    
    # Preparar polígonos urbanos
    gdf_export = gdf_cities[['CIUDAD', 'POB17', 'REGNAT', 'DEPARTAMENTO', 
                             'PROVINCIA', 'area_km2', 'densidad', 'geometry']].copy()
    
    # Consolidar redes de calles
    print("  Consolidando redes de calles...")
    all_edges = []
    
    for city_name, graph in all_graphs:
        if graph is not None:
            try:
                edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
                edges['ciudad_nombre'] = city_name
                all_edges.append(edges)
            except Exception as e:
                print(f"  ✗ Error exportando calles de {city_name}: {e}")
    
    # Guardar en un solo GeoPackage con dos layers
    gpkg_path = OUTPUT_DIR / 'data' / 'peru_sno.gpkg'
    
    # Layer 1: polígonos urbanos
    gdf_export.to_file(gpkg_path, layer='poligonos_urbanos', driver='GPKG')
    print(f"  ✓ Layer 'poligonos_urbanos' guardado")
    
    # Layer 2: red de calles
    if all_edges:
        gdf_streets = gpd.GeoDataFrame(pd.concat(all_edges, ignore_index=True))
        # Mantener solo columnas relevantes
        cols_to_keep = ['ciudad_nombre', 'geometry', 'bearing', 'length', 
                        'highway', 'name', 'oneway']
        cols_available = [c for c in cols_to_keep if c in gdf_streets.columns]
        gdf_streets = gdf_streets[cols_available]
        
        gdf_streets.to_file(gpkg_path, layer='red_calles', driver='GPKG')
        print(f"  ✓ Layer 'red_calles' guardado ({len(gdf_streets)} segmentos)")
    
    print(f"  ✓ peru_sno.gpkg creado con {2 if all_edges else 1} layers")
    
    # Resumen final
    print("\n" + "="*70)
    print("✓ PRE-PROCESAMIENTO COMPLETADO")
    print("="*70)
    print(f"\nArchivos generados en: {OUTPUT_DIR.absolute()}")
    print(f"\n  Fichas:          {len(list((OUTPUT_DIR / 'fichas').glob('*.png')))} archivos PNG")
    print(f"  Gráficos:        {len(list((OUTPUT_DIR / 'graficos').glob('*.png')))} archivos PNG")
    print(f"  Mapas:           {len(list((OUTPUT_DIR / 'mapas').glob('*.png')))} archivos PNG")
    print(f"  Datos JSON:      {len(list((OUTPUT_DIR / 'data').glob('*.json')))} archivos JSON")
    print(f"  GeoPackages:     {len(list((OUTPUT_DIR / 'data').glob('*.gpkg')))} archivos GPKG")
    print("\n¡Ahora puedes usar estos archivos en tu aplicación web!")

if __name__ == "__main__":
    main()