# ZDO 2022 - Hlubučková, Majer
## Zadání

Navrhněte a naprogramujte aplikaci pro detekci chirurgických nástrojů v čase Vstupem jsou cesty k videím z pohledu shora během šití prasečí nožičky Cílem je určit ```[x, y]``` pozice obarvených hrotů chirurgických nástrojů pro každý snímek Výstup (bude sloužit pro vyhodnocení na našich datech): json pro každou sekvenci obrázků bude ve formátu:
```python
annotation={

    "filename": ["vid1.mp4", "vid1.mp4", "vid1.mp4"...], # pth.parts[-1]
    
    "frame_id": [0,1,1,...],
    
    "object_id": [0,0,1],
    
    "x_px": [110, 110, 300], # x pozice obarvených hrotů v pixelech
    
    "y_px": [50, 50, 400],   # y pozice obarvených hrotů v pixelech
    
    "annotation_timestamp": [],   
}
```
vykreslení výstupů do videa:

centry hrotů vykreslit buď viditelným bodem nebo ohraničujícím boxem rozumné velikostí (např. 100x100 px)

- každý objekt označen jinou barvou
- Primární úloha: tracking needle holderu
- Bonusová úloha: tracking ostatních nástrojů

## Instalace

Pro použití GPU při běhu NN nejdříve
```
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```
Následně
```
pip install -r requirements.txt
```

Stáhnout předtrénovaný model NN z následujícího [linku](https://drive.google.com/file/d/19kXQhMYW0am4dWocBShoDSeKgFMUd4ED/view?usp=sharing) a náhrát do adresáře ```resources```

Lze spustit test v adresáři ```zdo2022/tests```, metoda ```predict()``` pro vytvoření anotací predikcí se nachází v ```zdo2022/main.py```

## Řešení

Problém byl řešen použitím neuronové sítě s architekturou Faster RCNN natrénovanou na našich datech spolu s OpenCV implementací CSRT trackeru.

### Trénovací data

Pro natrénování neuronové sítě bylo použito 1120 obrázků z trénovacího datasetu. Trénovací dataset sestává z obrázků poskytnutých a z obrázků augmentovaných. Data byla vybrána pseudonáhodně, kde jsme provedli ruční výběr množin obrázků (přibližně 4000 obrázků), z nich následně byla náhodně vybrána množina trénovacích dat. Tento výběr jsme provedli kvůli omezené výpočetní kapacitě tak, abychom měli dostatek dat pro všechny nástroje.

1120 obrázků následně bylo rozděleno v poměru 80:20 na trénovací a testovací data.

### Augmentace dat

Pro augmentaci dat byla využita knihovna Albumentations, byla vytvořena následující augmentation pipeline:

```python
transform = A.Compose([
        A.OneOf([![tframe_202_000719](https://user-images.githubusercontent.com/58400210/170769475-22947164-3f3c-4754-92da-5de8b08d038a.PNG)

            A.HorizontalFlip(p=0.7),
            A.Rotate(limit=5, interpolation=2, p=0.6)
        ], p=1),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=35, val_shift_limit=20, p=0.55),
            A.RGBShift(r_shift_limit=35, g_shift_limit=35, b_shift_limit=35, p=0.55)
        ], p=1),
        A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.0), p=0.5),
        A.RandomShadow(shadow_roi=(0.1, 0.1, 0.9, 0.9), shadow_dimension=7, p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=0.6),
            A.GaussianBlur(blur_limit=(3, 9), p=0.6),
        ], p=1)
    ], bbox_params=A.BboxParams(format='coco'), keypoint_params=A.KeypointParams(format='xy'))  
}
```

Z geometrických transformací používáme buď flip, nebo rotaci, z transformací jasů změnu buď hodnot RGB, nebo HSV. Následně je možnost přidat gaussovské rozmazání, nebo rozmazání simulující pohyb, zvýšit ostrost a přidat polygon stínu.

Příklad augmentovaného obrázku je vidět zde:

![tframe_220_000047](https://user-images.githubusercontent.com/58400210/170769037-d6c55854-af0e-4cc3-9991-75a2475386c6.PNG)

![tframe_202_000719](https://user-images.githubusercontent.com/58400210/170769494-df7ac161-080f-484d-952d-d7384e72e8cc.PNG)

### Predikce polohy hrotu

Jak již bylo zmíněno, pro predikci polohy hrotů používáme neuronovou síť doplněnou trackingem z OpenCV. Výstupům sítě přikládáme větší váhu, proto detekujeme nástroje na každém framu a pomocí těchto detekcí inicializujeme tracking. Pokud nástroj na framu není detekován pomocí sítě, je tracking použit k odhadu nejlepší polohy hrotu nástroje.

Zde je vidět příklad vykreslení výstupů predikce polohy hrotu do původního videa:

![priklad_nastroj](https://user-images.githubusercontent.com/58400210/170772246-b085efe6-8de2-4f68-901f-5a97ed959fe0.gif)


#### Chyby predikce

Systém predikce samozřejmě nedetekuje dokonale a OpenCV trackery mají také často tendenci k "přeskakování" na nejbližší pohyblivý objekt, což vede k chybám, které jsou časté hlavně při prudkých pohybech nástroje, při zminění nástroje z framu nebo při jeho okluzi. Při detekci neuronovou sítí se zbytečným chybám snažíme předcházet tím, že aplikujeme non-maximum suppression na redukci počtu predikcí a následně z redukovaného počtu predikcí bereme vždy jen jednu nejlepší pro každý objekt.

Příklad chyby predikce je vidět zde:

![priklad_chyba](https://user-images.githubusercontent.com/58400210/170773859-14246c5d-2d03-47bb-9542-354d2047543e.gif)

## Struktura kódu

Zde je výčet jednotlivých modulů a jejich stručný popis:

- ```main.py```: obsahuje metodu predict() ve třídě InstrumentTracker
- ```data_loader.py```: načtení poskytnutých anotací
- ```augmentations.py```: augmentace obrázků a bounding boxů, zápis anotací
- ```normalize_annotations.py```: sjednocení formátu augmentací pro účely tvorby datasetu k trénování NN
- ```datasets.py```: dataset vytvořený dle [následující stránky](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) pro fine-tuning NN
- ```finetune_nn.py```: skript pro fine-tuning NN a uložení natrénovaného modelu
- ```tracking.py```: funkce pro predikci polohy hrotu a vytvoření anotací
- ```show_results.py```: skript vykreslující polohy hrotů do videa
- ```transforms.py, engine.py, coco_eval.py, coco_utils.py, transforms.py, utils.py```: mírně upravené torchvision skripty tak, aby fungovalo trénování NN
