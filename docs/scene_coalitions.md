# SfM por escena, configuración y coalición

`run_scene_coalitions.py` lee `frames/<escena>/<plataforma>/...` y trabaja con
todas las imágenes de cada plataforma. No extrae videos ni calcula Shapley.
Para Car, cada `cam0`…`cam7` sigue siendo una cámara, pero Car es un único
participante en las coaliciones.

Para tres plataformas genera siete grupos: Car, Drone, Pedestrian,
Car+Drone, Car+Pedestrian, Drone+Pedestrian y Car+Drone+Pedestrian.
La coalición vacía no tiene reconstrucción.

## Ejecutar

Usa un entorno que tenga PyTorch, torchvision, NumPy, OpenCV, h5py, SciPy,
Pillow, tqdm y pycolmap. La integración fue probada con pycolmap 4.1.0.
En esta máquina existe el entorno `must3r`:

```bash
conda activate must3r

# Solo inspección: no importa los modelos, no escribe ni descarga nada
python run_scene_coalitions.py --scenes DavidHouse --configs sift sp-sg sp-lg --dry-run

# SIFT y SuperPoint + SuperGlue: 14 reconstrucciones en total
python run_scene_coalitions.py --scenes DavidHouse --configs sift sp-sg --top-k 20 --threads 2

# Agregar otra configuración; conserva features y pares previos
python run_scene_coalitions.py --scenes DavidHouse --configs sp-lg --top-k 20 --threads 2

# Omitir --scenes procesa todas las carpetas de escena en frames/
python run_scene_coalitions.py --configs sift --device cpu

# Detenerse al terminar una etapa; la siguiente ejecución reutiliza sus salidas
python run_scene_coalitions.py --scenes DavidHouse --configs sift sp-sg --until features
```

`--until` acepta `features`, `pairs`, `matches` y `sfm` (predeterminado).
`--frames-root` cambia la entrada; `--output-root` cambia las salidas.
`--platforms` permite cambiar los participantes, que deben existir y contener
imágenes en cada escena. El número de coaliciones crece como 2^n - 1.

## Configuraciones

- `sift`: SIFT de COLMAP, normalización L2, matching de vecinos más cercanos,
  ratio 0.8 y comprobación mutua. No requiere Kornia ni pesos aprendidos.
- `sp-sg`: SuperPoint + SuperGlue outdoor, usando las configuraciones locales de HLoc.
- `sp-lg`: comparte los mismos SuperPoint con sp-sg y usa LightGlue.
- `mast3r`: correspondencias directas entre imágenes con MASt3R original.
- `mast3r-aerialmd`: el mismo adaptador con el checkpoint ajustado en AerialMD.

Global por defecto: NetVLAD. `--global-feature` también admite `openibl`,
`megaloc` y `dir`, según extractores/dependencias/pesos disponibles en HLoc.
Los modelos globales pueden descargar pesos al ejecutarse por primera vez.

Para habilitar SuperPoint y SuperGlue en este checkout:

```bash
git submodule update --init --recursive third_party/SuperGluePretrainedNetwork
```

Para LightGlue, instalar en el mismo entorno:

```bash
python -m pip install git+https://github.com/cvg/LightGlue.git
```

Actualmente faltan ese submódulo y LightGlue en el entorno inspeccionado.
El pipeline comprueba estas dependencias antes de empezar una extracción larga.

## Retrieval y reutilización

Cada imagen actúa como consulta. Se buscan sus top-k vecinos globales solo
entre las imágenes de su propia coalición, excluyéndose a sí misma. Se usa
similitud coseno y se eliminan pares duplicados/inversos. No se filtran los
pares del grupo completo para construir los grupos pequeños: eso podría
privarlos de vecinos que sí obtendrían al hacer retrieval por separado.

La búsqueda se hace en CPU por bloques (`--query-batch 32`,
`--database-batch 512`); no se guarda la matriz N x N completa.

Las features globales se extraen una vez por escena y configuración global.
Las locales se extraen una vez por escena y configuración local. SuperPoint
se comparte entre SuperGlue y LightGlue. Para cada matcher se toma la unión
de pares de las siete coaliciones y se calcula cada par una sola vez.

Los resultados de diferentes matchers no se mezclan: SIFT/NN, SP/SG y SP/LG
necesitan sus propios matches. Cada reconstrucción usa únicamente su lista
de imágenes y sus pares, aunque los HDF5 compartidos contengan toda la escena.
La verificación geométrica, la base de datos y SfM se ejecutan por coalición.

Al repetir el comando se reutilizan features y matches completos y modelos
terminados. Se reparan entradas HDF5 incompletas. Cambiar top-k vuelve a hacer
retrieval y SfM pero reutiliza los matches de pares ya presentes. Cambiar el
matcher conserva las features. Cambiar `--seed` vuelve a hacer SfM.

Se separan cachés por rutas/tamaños/fechas de las imágenes, configuración,
versión de compatibilidad de implementación y versiones de PyTorch/pycolmap.
La versión de compatibilidad se conserva al añadir opciones de CLI o MVS;
no se invalida todo por editar el script. Se mantiene la identidad del pipeline
previo a esta ampliación, incluida su caché NetVLAD existente. Si se cambia la
semántica interna de extracción/matching disperso, debe actualizarse
`CACHE_IMPLEMENTATION` para no reutilizar resultados incompatibles. Si cambian las imágenes se
crea otro snapshot; los anteriores se conservan. No se hace hash del contenido
de todas las imágenes: no modifiques sus bytes conservando tamaño y fecha.
Los pesos personalizados deben distinguirse en la configuración o usar otro
`--output-root`. No se importan automáticamente cachés de pipelines anteriores.

## Añadir pares secuenciales

```bash
python run_scene_coalitions.py --scenes DavidHouse --configs sift sp-sg --top-k 20 --sequential-window 5
```

Combina los pares de retrieval con los pares de cada imagen y sus siguientes
cinco muestras dentro del mismo video y cámara. No duplica pares compartidos
entre ambas estrategias. `--sequential-window 0` (predeterminado) lo desactiva.
La ventana cuenta imágenes extraídas, no frames del video original ni segundos.

Agrupa por carpeta y prefijo del video y ordena el índice final numéricamente;
reconoce los nombres `video_mp4__000001.jpg` generados por el extractor. No une
el final de un clip con el siguiente, ni distintas cámaras o plataformas.
Los nombres sin índice numérico final o índices repetidos se rechazan cuando
se activa esta opción. Para nombres personalizados, el prefijo debe identificar
el video de origen dentro de cada carpeta.

La ventana forma parte de la configuración de pares/SfM. Cambiarla entre
ejecuciones de esta versión conserva las features y los matches ya calculados;
solo se calculan los nuevos pares necesarios y las nuevas reconstrucciones.

## Salidas

```text
outputs/coalitions/DavidHouse/<snapshot>/
  dataset.json
  features/
    global-<id>.h5
    local-<id>.h5
    ... configuraciones .json
  pairs/<retrieval-id>/
    Car.txt
    Car+Drone.txt
    ...
  matches/<local-matcher-id>.h5
  reconstructions/<preset-id>/
    config.json
    summary.json
    Car/
      result.json
      attempt_<id>/
        database.db
        cameras.bin
        images.bin
        points3D.bin
        rigs.bin
        frames.bin
    Car+Drone/
    ...
```

HLoc conserva el mayor modelo por número de imágenes registradas como modelo
principal; puede haber otros componentes bajo `models/`. `result.json` refleja
solo el principal. `no_model` significa que COLMAP terminó sin modelo válido;
`error` significa un fallo de ejecución y no se considera un resultado válido.
Un no_model se conserva; usa otra semilla o top-k para otro intento. Los errores
se reintentan en otra carpeta sin borrar intentos anteriores. El pipeline se
detiene ante errores para evitar mezclar una ejecución parcial con una completa.

`--camera-mode PER_FOLDER` comparte intrínsecos por carpeta (Car/cam0, Drone,
etc.). Exige que esas imágenes correspondan a la misma lente/zoom/resolución.
Usa subcarpetas distintas si cambian, o `--camera-mode PER_IMAGE`. No se impone
un rig sincronizado ni se mezclan todas las cámaras del carro en una sola.

## Recursos y reproducibilidad

Procesa una escena, configuración y coalición a la vez. Matching es secuencial
sin la cola multiproceso de HLoc. `--threads` limita PyTorch/OpenCV y, en Linux,
la afinidad del proceso y sus hijos para cubrir también operaciones nativas.
`--device cpu` desactiva CUDA para esta ejecución. Las features se guardan en
disco; el matching aún requiere memoria cuadrática en el número de keypoints
por par. Puedes reducir `--max-keypoints` (4096) y `--resize-max` (1024).

Se fija semilla, pero CUDA, orden de operaciones y SfM pueden introducir
variaciones; no se garantiza igualdad bit a bit entre máquinas. Todos los
grupos usan la misma receta; no se equilibra la cantidad de imágenes por
plataforma. No se interpreta el número de puntos como precisión geométrica.

Pruebas:

```bash
python -m unittest discover -s tests -p test_scene_coalitions.py -v
```


## MASt3R y MASt3R + AerialMD

```bash
python run_scene_coalitions.py --scenes DavidHouse \
  --configs mast3r mast3r-aerialmd --top-k 20 --sequential-window 10 --threads 2
```

Reutiliza `hloc/matchers/mast3r.py` y los dos checkpoints de tu integración.
La importación del adaptador fue verificada en el entorno `must3r`. Los pesos
pueden descargarse si no existen; no se ejecutó inferencia real con ellos en
las pruebas de esta ampliación.

No se ejecuta SuperPoint/SIFT antes de MASt3R. El modelo recibe cada par de
imágenes y produce coordenadas y confianza. Se adapta el preprocesamiento de
HLoc de [0,1] a [-1,1], como la normalización original de MASt3R. La resolución
máxima de este adaptador es 512 px; `--resize-max` controla los extractores
locales dispersos, no este tamaño.

`dense_raw/<id>.h5` guarda coordenadas originales y confianza por par. La unión
de pares de todas las coaliciones se infiere una sola vez por checkpoint.
Cambiar top-k o ventana secuencial añade solo los pares faltantes. Un par vacío
válido también queda registrado y no se vuelve a calcular. Las entradas
interrumpidas se reparan individualmente.

`dense_assembled/<id>/<coalition-pairs-id>/` contiene features y matches aptos
para COLMAP. Se forman tracks únicamente con pares de esa coalición, mediante
celdas de 1 px y selección de keypoints por confianza acumulada. Los duplicados
se resuelven por confianza con correspondencia uno a uno. Este ensamblado es
propio del pipeline; no es la optimización global 3D nativa de MASt3R. Su límite
es `--max-keypoints`. Si cambian los pares o ese límite, se vuelve a ensamblar
sin repetir la inferencia neuronal de los pares ya disponibles.

Con estos presets, `--until features` obtiene solo features globales;
`--until matches` infiere y ensambla, y `--until sfm` reconstruye las coaliciones.
No se importan automáticamente HDF5 de `run_pipeline_mast3r.py`, pues su
preprocesamiento y agregación pueden diferir. Los modelos COLMAP ya generados
por ese pipeline sí pueden pasarse directamente a MVS.

## Retomar y ejecutar MVS después

Repite el mismo comando de SfM: la reanudación es automática. No necesitas
`--resume`. Se omiten las imágenes, pares y modelos terminados. Si se interrumpe
el mapper durante una coalición, esa reconstrucción se reinicia en otro intento;
no se promete reanudar el optimizador exactamente en la iteración interrumpida.
Las demás coaliciones terminadas se conservan. No lances otra ejecución sobre
la misma escena mientras el proceso anterior siga activo: el lock lo impide.

MVS es independiente y no vuelve a extraer features ni a ejecutar SfM:

```bash
# Ver los modelos terminados que se usarán
python run_mvs.py --sfm-root outputs/coalitions/DavidHouse --dry-run

# Densificar todos los modelos terminados encontrados
python run_mvs.py --sfm-root outputs/coalitions/DavidHouse \
  --max-image-size 1024 --cache-gb 2 --threads 2 --gpu-index 0

# Filtrar configuración y coalición
python run_mvs.py --sfm-root outputs/coalitions/DavidHouse \
  --configs mast3r mast3r-aerialmd --coalitions Car+Drone+Pedestrian

# Un modelo externo, por ejemplo del pipeline anterior de AerialMD
python run_mvs.py --model-path /ruta/al/modelo_colmap \
  --image-path /ruta/a/las/imagenes --output /ruta/nueva/mvs
```

Solo se descubren modelos con `result.json` terminado; `--sfm-root` puede
apuntar a una escena, snapshot o ejecución concreta para evitar densificar
versiones antiguas. El modelo principal de cada coalición se procesa en serie.
Para otros modelos/componentes, usa `--model-path` y la raíz de imágenes que
coincida con los nombres registrados en ese modelo.

Usa los comandos COLMAP `image_undistorter`, `patch_match_stereo` y
`stereo_fusion`; el resultado es `workspace/fused.ply`. El binario inspeccionado
está compilado con CUDA. PatchMatch requiere una GPU CUDA disponible en esta
instalación; `--threads` limita CPU, no sustituye la GPU. No se crea una malla.

MVS guarda `mvs_state.json` y logs por etapa. Al repetirlo omite etapas
terminadas con salidas válidas. Un PatchMatch interrumpido conserva los mapas
completos y aparta los truncados antes de continuar. La fusión interrumpida se
vuelve a ejecutar. Cambiar parámetros/modelo/imágenes crea por defecto otra
carpeta; si se fija `--output`, rechaza configuraciones incompatibles.

Los modelos SfM originales no se modifican. La salida automática está en una
carpeta `mvs/<modelo-id>` junto al modelo de entrada. Se limita resolución,
caché y CPU, pero la memoria total incluye buffers adicionales de COLMAP.

Las pruebas de MVS simulan la ejecución de comandos y fallos para verificar
reanudación y validación de archivos; no se lanzó la densificación GPU real.
Referencia: https://colmap.github.io/cli.html

## Dispositivo de SIFT

SIFT usa `pycolmap.FeatureExtractor.create()` y hereda `--device`. Puede
sobrescribirse con `--sift-device cpu|cuda|auto`. Con `auto`, SIFT solo elige
GPU si pycolmap está compilado con CUDA y hay GPU disponible. CUDA explícito
falla con una explicación si no puede cumplirse; no cae silenciosamente a CPU.

El entorno `must3r` tiene ahora `pycolmap-cuda12==4.1.0`, con
`pycolmap.has_cuda=True`. Se verificaron extracción SIFT GPU y PyTorch CUDA.
Para elegir CPU en SIFT de forma explícita y mantener GPU en modelos/matching:

```bash
python run_scene_coalitions.py --scenes DavidHouse --configs sift \
  --top-k 20 --sequential-window 10 --threads 2 --device cuda --sift-device cpu
```

El binario CLI `colmap` y el paquete Python `pycolmap` son instalaciones
independientes. Que el primero tenga CUDA no habilita CUDA en el segundo.
Las features ya extraídas se conservan al cambiar dispositivo; la GPU/CPU
puede producir pequeñas diferencias numéricas en las nuevas features.


Instalación GPU validada en `must3r`: `pycolmap-cuda12==4.1.0` con
`cuda-toolkit[cudart,curand]==12.6.3`, runtime 12.6.77 y cuRAND 10.3.7.77.
Se conserva PyTorch 2.7.0+cu126. El paquete CUDA reemplaza a `pycolmap` CPU;
no instales ambos simultáneamente. El `requirements.txt` general todavía
incluye `pycolmap` CPU: no lo reinstales encima de este entorno sin adaptar
esa dependencia. La versión importada sigue siendo 4.1.0 y conserva las claves
de caché actuales. Fuente: https://colmap.github.io/pycolmap/index.html
