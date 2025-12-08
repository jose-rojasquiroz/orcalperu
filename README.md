Orientación de la red de calles de las principales ciudades del Perú, identificadas por el Instituto Nacional de Estadística e Informática (INEI).

En el repositorio encontrarán:

1. `dataprep_peru_sno.ipynb`: Un cuaderno en el que uno varios archivos previos en uno sólo, que tiene el polígono urbano de cada una de las 91 ciudades principales identificadas por INEI, con su respectiva población y la región natural a la que pertenece. 
2. `pu-principalesciudades-WGS84.gpkg`: El archivo vectorial obtenido con el cuaderno anterior.
3. `peru_sno.ipynb`: Un cuaderno en el que se utiliza el paquete `osmnx` para obtener la orientación de las calles. En él se hace el cálculo para las 91 ciudades y también para las cinco más pobladas de cada región natural.

Si deseas replicar por completo el análisis es necesario revisar los archivos `uv.lock`, `.python-version` y `pyproject.toml`. Con [uv](https://github.com/astral-sh/uv), el gestor de librerías de Python se pueden usar las mismas versiones de las librerías usadas para los resultados mostrados. 

El análisis está basado en el trabajo de [Boeing (2021)](https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0189-1). 
Para mayor información sobre el uso de `omsnx` recomiendo leer este [artículo](https://geoffboeing.com/2016/11/osmnx-python-street-networks/). 
________________________

Para publicaciones que usen estos datos, por favor cítalos de la siguiente manera: "Rojas-Quiroz, J. (2024). Orientación de la red de calles de las principales ciudades del Perú (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.13918305".
