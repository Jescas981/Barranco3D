# Target point
TARGET_LAT = -12.143334999999539
TARGET_LON = -77.02316199999836

```
    python download_mapilary_dataset.py --lat -12.143334999999539 --lon -77.02316199999836 --radius 50 --min-distance 1 --max-images 500 --limit 2000
```

## Extraer frames de toda una escena

Requiere Python 3.9+ y `ffmpeg`/`ffprobe`. No requiere OpenCV.

```bash
python3 extract_colmap_frames.py DavidHouse --fps 2 --dry-run
python3 extract_colmap_frames.py DavidHouse --fps 2
```

Busca `datasets/DavidHouse`; admite también la carpeta existente
`dataset/DavidHouse`. Puedes pasar cualquier ruta explícita como entrada.
Extrae todas las plataformas por defecto, reflejando sus subcarpetas:

```text
datasets/DavidHouse/           frames/DavidHouse/
  Car/cam0/*.mp4                 Car/cam0/*.jpg
  Car/cam1/*.mp4                 Car/cam1/*.jpg
  ...                           ...
  Car/cam7/*.mp4                 Car/cam7/*.jpg
  Drone/*.MP4                    Drone/*.jpg
  Pedestrian/*.mp4               Pedestrian/*.jpg
                                _colmap/
                                  extraction.json
                                  image_list.txt
                                  run_colmap.sh
                                  README.md
                                  sparse/
```

Los nombres incluyen el video y el índice de muestra para evitar colisiones.
Se conserva resolución original. `--format png` exporta sin pérdida adicional;
`--output frames/DavidHouse_05fps` permite guardar otra extracción por separado.
La carpeta de salida debe estar vacía o ser nueva. `--groups` es un filtro
opcional; sin él se procesa toda la escena.

Procesa videos en serie con un hilo (`--threads`). El muestreo usa timestamps
de cada clip; no sincroniza cámaras. Los FPS deben ser positivos y no superar
la tasa media de los videos; en video variable pueden repetirse imágenes al
cubrir huecos temporales.

Para reconstruir: `sh frames/DavidHouse/_colmap/run_colmap.sh`.
COLMAP usa `frames/DavidHouse` como raíz de imágenes y crea su base de datos
bajo `_colmap`. Se comparte una cámara por carpeta, por lo que los videos de
cada carpeta deben tener la misma resolución y configuración óptica. Si cambia
la lente o zoom, sepáralos en subcarpetas antes de extraer.
COLMAP no se ejecuta automáticamente. Su matcher exhaustivo puede ser costoso.

## Reconstrucciones de coaliciones

[Pipeline por escena y configuración](docs/scene_coalitions.md): extrae features
globales/locales compartidas, hace retrieval por coalición, reutiliza matches
y genera las siete reconstrucciones de Car, Drone y Pedestrian. No calcula Shapley.

```bash
python run_scene_coalitions.py --scenes DavidHouse --configs sift sp-sg --dry-run
python run_scene_coalitions.py --scenes DavidHouse --configs sift sp-sg --threads 2
```

Revisa las dependencias y los comandos de instalación en la documentación antes
de ejecutar SuperPoint/SuperGlue o LightGlue.


También se admiten los matchers densos de tu integración:

```bash
python run_scene_coalitions.py --scenes DavidHouse --configs mast3r mast3r-aerialmd --sequential-window 10
```

La reanudación es automática. Para ejecutar MVS sobre SfM terminados, en otro momento:

```bash
python run_mvs.py --sfm-root outputs/coalitions/DavidHouse --threads 2 --max-image-size 1024
```

Consulta [configuración, cachés densas y MVS](docs/scene_coalitions.md) para detalles.
