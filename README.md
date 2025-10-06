([Français](#plugin-treeaibox-pour-cloudcompare))

# <img src="https://github.com/NRCan/TreeAIBox/blob/b0918d50b1343d62e1b518c29cd50d38f801b359/treeaibox-header.bmp" alt="treeaibox_logo" width="60"/> TreeAIBox CloudCompare Plugin

A CloudCompare Python plugin providing a unified web-style GUI for a suite of LiDAR processing modules targeting forest and tree analysis.

It enables forestry practitioners and researchers to interactively process 3D LiDAR data within the open-source CloudCompare software.

This aligns with the idea that agentic AI should first break down tasks into sub-questions and then fine-tune smaller, specialized vision models—an approach that can be both more effective and more cost-efficient than relying solely on large language models.

## 📖 Overview

TreeAIBox brings together four core LiDAR-processing workflows in a single GUI:

- **TreeFiltering**  
  Supervised deep-learning filtering to separate understory and overstory points.
  
- **TreeisoNet**  
  End-to-end crown segmentation pipeline (StemCls → TreeLoc → TreeOff → CrownOff3D), allowing manual editing.
  
- **WoodCls**  
  3D stem & branch classification on TLS data.
  
- **QSM**  
  Plot-level skeletonization and export of tree structure to XML/OBJ.

- **UrbanFiltering**  
  Supervised deep learning–based filtering to classify urban scenes into seven categories: 1 = ground, 2 = vegetation, 3 = vehicles (cars+trucks), 4 = powerlines, 5 = fences, 6 = poles, 7 = buildings.


## 🚀 Features

- **20+ pretrained AI models**  
  Downloadable from a remote server; mostly lightweight or distilled versions, fine-tuned with carefully annotated datasets.
  
- **3D targeted**  
  Operates directly on raw 3D point clouds—no CHM or raster inputs—using voxel-based AI architectures for both training and inference.
  
- **Sensor, scene, and resolution options**  
  Supports TLS, ALS, and UAV LiDAR across boreal, mixedwood, reclamation, and urban forest types.
  
- **GPU acceleration toggle**  
  Runs on either GPU (CUDA) or CPU for flexibility.
  
- **Web-style UI framework**  
  Features resizable windows and modular UI components.
  
- **Interactive parameter controls**  
  Allow certain result customization using adjustable parameters.
  
- **Open source**  
  Fully Python-based (except for pretrained model files); outputs include scalar fields, point clouds, and exportable files.
  
- **Windows installer**  
  Automatically installs required packages and registers the main script as a Python plugin.

## 🛠️ Installation

### 1. Via Windows installer (Suggested)

A ready-to-run online installer is provided. Ensure **internet access** is enabled:

1. **Install CloudCompare**
   
   Download and install **CloudCompare v2.14.alpha** (latest release) from
   [https://cloudcompare-org.danielgm.net/release/](https://cloudcompare-org.danielgm.net/release/)

2. **Download the TreeAIBox Installer**
   
   Get **TreeAIBox\_Plugin\_Installer\_v1.0.exe** from our releases page:
   [https://github.com/NRCan/TreeAIBox/releases](https://github.com/NRCan/TreeAIBox/releases)

3. **Run the Installer**

   * Right-click **TreeAIBox\_Plugin\_Installer\_v1.0.exe** and choose **Run as administrator**.
   * Follow the on-screen prompts. By default, the installer will detect your CloudCompare folder (e.g., `%PROGRAMFILES%\CloudCompare`) from the registry.

   **What Installer does:**

   * Copies all Python scripts, UI files, images, and modules into:

     ```
     …\CloudCompare\plugins\Python\Plugins\TreeAIBox\
     ```
   * Generates a helper batch script to detect your NVIDIA GPU and install the matching PyTorch wheel.
   * Launches `pip` to install required Python packages (e.g., PyQt6, torch, requests).

4. **Finish Up**

   Once installation completes, restart CloudCompare and launch TreeAIBox from the Python console.

> **Note:** The NSIS script (`CloudCompare_Python_Plugin.nsi`) can be edited if you need to customize install paths or package versions.

### 2. Alternatively, manual (Git + pip)

```bash
cd %PROGRAMFILES%\CloudCompare\plugins\Python\Plugins
git clone https://github.com/NRCan/TreeAIBox.git TreeAIBox
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install PyQt6 PyQt6-WebEngine requests numpy_indexed timm numpy_groupies cut_pursuit_py circle_fit scikit-learn scikit-image
```
Under CloudCompare’s Script Register menu, click Add Script, then browse to and select TreeAIBox.py to register it.

## ▶️ Usage

In CloudCompare, under Script Register, click the TreeAIBox.

![image](https://github.com/user-attachments/assets/f76865eb-ba91-4516-85b7-42c08d10ccb5)

Then select a point cloud, pick your module tab, choose/download a model, adjust settings, and click **Apply**.

![TreeFiltering](https://github.com/user-attachments/assets/76b9dd32-a7c9-44de-9a92-f9f55806e9b7)
![TreeIsoNet](https://github.com/user-attachments/assets/1457ecb3-5fa3-4e00-b6b9-441bac9a1368)
![WoodCls](https://github.com/user-attachments/assets/7b9adfc1-76f1-4af4-9a54-e673cbf3ddc0)
![UrbanFiltering](https://github.com/user-attachments/assets/8b98f467-468d-4732-ab70-1cbaf420ebce)
![QSM](https://github.com/user-attachments/assets/d2d8b93d-4823-477e-be93-490d2c04a6a0)

When isolating ALS individual trees, set an appropriate voxel resolution (e.g., 0.8 m horizontal by 2.0 m vertical) to optimize tree detection on TreeisoNet.

## ⚙️ Configuration

- **`model_zoo.json`** lists available model names.  
- Logs & outputs in `C:\Users\USERNAME\AppData\Local\CloudCompare\TreeAIBox\`.

The table below summarizes the voxel resolution and GPU memory used by the current AI models, categorized by sensor type, task, component, and scene:

<table>
  <thead>
    <tr>
      <th align="center">Sensor</th>
      <th align="center">Task</th>
      <th align="center">Component</th>
      <th align="center">Scene</th>
      <th align="center">Resolution</th>
      <th align="center">VRAM</th>
    </tr>
  </thead>
  <tbody>
    <!-- ALS Classification -->
    <tr>
      <td align="center" rowspan="4"><strong>ALS (or UAV without stems)</strong></td>
      <td align="center" rowspan="4">Classification</td>
      <td align="center" rowspan="3">Vegetation layer</td>
      <td align="center">Mountainous</td>
      <td align="center">80 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Regular</td>
      <td align="center">50 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Wellsite</td>
      <td align="center">15 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Urban layers</td>
      <td align="center">Urban</td>
      <td align="center">30 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <!-- UAV Classification -->
    <tr>
      <td align="center" rowspan="2"><strong>UAV (with stems)</strong></td>
      <td align="center" rowspan="2">Classification</td>
      <td align="center">Vegetation layer</td>
      <td align="center">Regular</td>
      <td align="center">12 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Stems</td>
      <td align="center">Mixedwood</td>
      <td align="center">8 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <!-- TLS Classification -->
    <tr>
      <td align="center" rowspan="7"><strong>TLS</strong></td>
      <td align="center" rowspan="7">Classification</td>
      <td align="center">Vegetation layer</td>
      <td align="center">Regular</td>
      <td align="center">8 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center" rowspan="3">Stems</td>
      <td align="center" rowspan="3">Boreal</td>
      <td align="center">10 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">4 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">20 cm</td>
      <td align="center">8 GB</td>
    </tr>
    <tr>
      <td align="center">Stems</td>
      <td align="center">Regular</td>
      <td align="center">4 cm</td>
      <td align="center">12 GB</td>
    </tr>
    <tr>
      <td align="center" rowspan="2">Stems + branches</td>
      <td align="center" rowspan="2">Regular</td>
      <td align="center">4 cm</td>
      <td align="center">2 GB</td>
    </tr>
    <tr>
      <td align="center">2.5 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <!-- ALS Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>ALS (or UAV without stems)</strong></td>
      <td align="center" rowspan="2">Clustering</td>
      <td align="center">Tree tops</td>
      <td align="center">Wellsite</td>
      <td align="center">10 cm</td>
      <td align="center">4 GB</td>
    </tr>
    <tr>
      <td align="center">Tree segments</td>
      <td align="center">Wellsite</td>
      <td align="center">10 cm</td>
      <td align="center">4 GB</td>
    </tr>
    <!-- UAV Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>UAV (with stems)</strong></td>
      <td align="center" rowspan="2">Clustering</td>
      <td align="center">Tree bases</td>
      <td align="center">Mixedwood</td>
      <td align="center">10 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Tree segments</td>
      <td align="center">Mixedwood</td>
      <td align="center">15 cm</td>
      <td align="center">4 GB</td>
    </tr>
    <!-- TLS Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>TLS</strong></td>
      <td align="center" rowspan="2">Clustering</td>
      <td align="center">Tree bases</td>
      <td align="center">Boreal</td>
      <td align="center">10 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Tree segments</td>
      <td align="center">Boreal</td>
      <td align="center">15 cm</td>
      <td align="center">4 GB</td>
    </tr>
  </tbody>
</table>


## 🗂️ Folder structure

```
TreeAIBox-main
│   TreeAIBox_Plugin_Installer.exe                                  # Windows installer for the plugin
│   CloudCompare_Python_Plugin.nsi                                  # Configuration of the plugin installer
│   treeaibox-header.bmp                                            # Installer icon
│   treeaibox-welcome.bmp                                           # Installer icon
│   dl_visualization.svg                                            # The main DL network structure illustration
│   LICENSE.txt                                                     # License file
│   model_zoo.json                                                  # List of available trained DL model file names
│   README.md                                                       # README
│   TODO.md                                                         # To-do list
│   TreeAIBox.py                                                    # Main python program of TreeAIBox
│   treeaibox_logo.ico                                              # Plugin logo
│   treeaibox_ui.html                                               # Main GUI (web view)
├───img                                                             # Icons and images used by the plugin GUI
└───modules                                                         # Submodules of TreeAIBox
    ├───filter                                                      # TreeFiltering and WoodCls modules
    │       componentFilter.py                                      # Functions of filtering tree layer, branch, and stem components
    │       createDTM.py                                            # Functions of creating DTM grid points based on the filtered tree and ground layers
    │       *.json                                                  # Definition of DL model parameters
    │       vox3DESegFormer.py                                      # DL model structure (version 2)
    │       vox3DSegFormer.py                                       # DL model structure (version 1)
    │       __init__.py
    │
    ├───qsm                                                         # QSM module
    │       applyQSM.py                                             # Functions of skeletonizing and reconstructing 3D tree geometries
    │       __init__.py
    │
    └───treeisonet
            cleanSmallerClusters.py                                 # Functions of roughly removing small clusters based on point number
            treeLoc.py                                              # Functions of extracing stem locations or tree tops
            treeOff.py                                              # Functions of clustering crown points to the extracted tree tops (for the stem-invisible scene)
            stemCluster.py                                          # Functions of clustering stem points to the stem locations based on the shortest path rule
            crownCluster.py                                         # Functions of clustering crown points to the segmented stem points based on the shortest path rule
            crownOff.py                                             # Functions of clustering crown points to the segmented stem points based on deep learning clustering
            treeStat.py                                             # Functions of extracting individual-tree and plot-level statistics
            *.json                                                  # Definition of DL model parameters
            vox3DSegFormerDetection.py                              # DL model structure of tree location detection
            vox3DSegFormerRegression.py                             # DL model structure of point offset regression
            __init__.py
```

## 🤝 How to Contribute

1. Fork → feature branch → PR.  
2. Follow existing style and add tests as needed.

See [CONTRIBUTING.md](CONTRIBUTING.md)


## 📄 License

Unless otherwise noted, the source code of this project is covered under Crown Copyright, Government of Canada, and is distributed under the [Creative Commons Attribution-NonCommercial 4.0 International](LICENSE.txt).

The Canada wordmark and related graphics associated with this distribution are protected under trademark law and copyright law. No permission is granted to use them outside the parameters of the Government of Canada's corporate identity program. For more information, see [Federal identity requirements](https://www.canada.ca/en/treasury-board-secretariat/topics/government-communications/federal-identity-requirements.html).


Developed by Zhouxin Xi, tested by Charumitha Selvaraj

*Born from over a decade of LiDAR research with support from dedicated collaborators.*

![image](https://github.com/user-attachments/assets/2cac174d-f874-4a4a-bc4d-93c6ee9d4905)

---
<a id="plugin-treeaibox-pour-cloudcompare"></a>
# <img src="https://github.com/user-attachments/assets/2ac22555-917d-45ab-873d-120618e66e76" alt="treeaibox_logo" width="32"/> Plugin TreeAIBox pour CloudCompare

Un plugin Python pour CloudCompare offrant une interface graphique de style web unifiée pour une suite de modules de traitement LiDAR dédiés à l’analyse forestière et arboricole.

Il permet aux praticiens et aux chercheurs forestiers de traiter de manière interactive des données LiDAR 3D au sein du logiciel open source CloudCompare.

Cela s’inscrit dans l’idée que l’IA agentielle devrait d’abord décomposer les tâches en sous-questions, puis affiner des modèles de vision plus petits et spécialisés — une approche à la fois plus efficace et plus économique que de se fier uniquement aux grands modèles de language.

## 📖 Vue d’ensemble

TreeAIBox regroupe quatre flux de travail LiDAR essentiels dans une seule interface :

- **TreeFiltering**  
  Filtrage supervisé par apprentissage profond pour séparer les points de sous-étage et de sur-étage.

- **TreeisoNet**  
  Pipeline de segmentation de la couronne de bout en bout (StemCls → TreeLoc → TreeOff → CrownOff3D), avec possibilité d’édition manuelle.

- **WoodCls**  
  Classification 3D des tiges et des branches sur données TLS.

- **QSM**  
  Squelettisation au niveau de la parcelle et export de la structure des arbres au format XML/OBJ.
  
- **UrbanFiltering**
  Filtrage supervisé basé sur l’apprentissage profond pour classer les scènes urbaines en sept catégories : 1 = sol, 2 = végétation, 3 = véhicules (voitures + camions), 4 = lignes électriques, 5 = clôtures, 6 = poteaux, 7 = bâtiments.

## 🚀 Fonctionnalités

- **Plus de 20 modèles IA préentraînés**  
  Téléchargeables depuis un serveur distant ; versions légères ou distillées, ajustées sur des jeux de données annotés avec soin.

- **3D ciblé**  
  Fonctionne directement sur des nuages de points 3D bruts — pas d’entrée CHM ou raster — utilisant des architectures IA basées sur les voxels pour l’entraînement et l’inférence.

- **Options de capteur, de scène et de résolution**  
  Prend en charge le LiDAR TLS, ALS et UAV pour les forêts boréales, mixtes, en restauration et urbaines.

- **Bascule d’accélération GPU**  
  Exécution sur GPU (CUDA) ou CPU pour plus de flexibilité.

- **Cadre UI de type web**  
  Fenêtres redimensionnables et composants UI modulaires.

- **Contrôles interactifs de paramètres**  
  Personnalisation des résultats via des paramètres ajustables.

- **Open source**  
  Entièrement basé en Python (à l’exception des fichiers de modèles préentraînés) ; sorties : champs scalaires, nuages de points et fichiers exportables.

- **Installateur Windows**  
  Installation automatique des paquets requis et enregistrement du script principal en tant que plugin Python.

## 🛠️ Installation

### 1. Via l’installateur Windows (recommandé)

Un installateur en ligne prêt à l’emploi est fourni. Assurez-vous que **l’accès Internet** est activé :

1. **Installer CloudCompare**

   Téléchargez et installez **CloudCompare v2.14.alpha** (dernière version) depuis
   [https://cloudcompare-org.danielgm.net/release/](https://cloudcompare-org.danielgm.net/release/)

2. **Télécharger l’installateur TreeAIBox**

   Récupérez **TreeAIBox\_Plugin\_Installer\_v1.0.exe** depuis notre page de releases :
   [https://github.com/NRCan/TreeAIBox/releases](https://github.com/NRCan/TreeAIBox/releases)

3. **Exécuter l’installateur**

   * Faites un clic droit sur **TreeAIBox\_Plugin\_Installer\_v1.0.exe** et choisissez **Exécuter en tant qu’administrateur**.
   * Suivez les instructions à l’écran. Par défaut, l’installateur détecte le dossier CloudCompare (ex. `%PROGRAMFILES%\CloudCompare`) via le registre.

   **Ce que fait l'installateur**

   * Copie tous les scripts Python, fichiers UI, images et modules dans :

     ```
     …\CloudCompare\plugins\Python\Plugins\TreeAIBox\  
     ```
   * Génère un script batch d’aide pour détecter votre GPU NVIDIA et installer la roue PyTorch correspondante.
   * Lance `pip` pour installer les paquets Python requis (par ex. PyQt6, torch, requests).

4. **Finalisation**

   Une fois l’installation terminée, redémarrez CloudCompare et lancez TreeAIBox depuis la console Python.

> **Remarque :** Le script NSIS (`CloudCompare_Python_Plugin.nsi`) peut être modifié si vous devez personnaliser les chemins d’installation ou les versions des paquets.


### 2. Alternative manuelle (Git + pip)

```bash
cd %PROGRAMFILES%\CloudCompare\plugins\Python\Plugins
git clone https://github.com/NRCan/TreeAIBox.git TreeAIBox
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install PyQt6 PyQt6-WebEngine requests numpy_indexed timm numpy_groupies cut_pursuit_py circle_fit scikit-learn scikit-image
````

Dans CloudCompare, enregistrez `TreeAIBox.py` via le bouton « Add Script » du menu Script Register.

## ▶️ Utilisation

Dans CloudCompare, sous **Script Register**, cliquez sur **TreeAIBox**.

![capture d’écran TreeFiltering](https://github.com/user-attachments/assets/ee4f3558-6535-448b-8279-d0a4cff2158f)

Sélectionnez ensuite un nuage de points, choisissez l’onglet du module souhaité, sélectionnez/téléchargez un modèle, ajustez les paramètres, puis cliquez sur **Apply**.

![capture d’écran TreeIsoNet](https://github.com/user-attachments/assets/d82e3bf6-8db1-49a4-a7c2-134ce4760fec)
![capture d’écran WoodCls](https://github.com/user-attachments/assets/2cf1288e-d9e8-4cf8-8251-4e8e2dcd17ec)
![capture d’écran QSM](https://github.com/user-attachments/assets/aa1a0bc8-febe-41d1-8bdb-f952970b5017)

Lors du traitement des données ALS, définissez une résolution de voxel appropriée (par exemple, 0,8 m horizontal sur 2,0 m vertical) pour optimiser la détection des arbres.

## ⚙️ Configuration

* **`model_zoo.json`** liste les noms de modèles disponibles.
* Journaux et sorties dans `C:\Users\USERNAME\AppData\Local\CloudCompare\TreeAIBox\`.

Le tableau ci-dessous résume la résolution voxel et la mémoire GPU utilisée par les modèles IA actuels, classés par type de capteur, tâche, composant et scène :

<table>
  <thead>
    <tr>
      <th align="center">Capteur</th>
      <th align="center">Tâche</th>
      <th align="center">Composant</th>
      <th align="center">Scène</th>
      <th align="center">Résolution</th>
      <th align="center">VRAM</th>
    </tr>
  </thead>
  <tbody>
    <!-- ALS Classification -->
    <tr>
      <td align="center" rowspan="4"><strong>ALS (ou UAV sans tiges)</strong></td>
      <td align="center" rowspan="4">Classification</td>
      <td align="center" rowspan="3">Couche de végétation</td>
      <td align="center">Montagneuse</td>
      <td align="center">80 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <tr>
      <td align="center">Régulière</td>
      <td align="center">50 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <tr>
      <td align="center">Site pétrolier</td>
      <td align="center">15 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <tr>
      <td align="center">Couches urbaines</td>
      <td align="center">Urbaine</td>
      <td align="center">30 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <!-- UAV Classification -->
    <tr>
      <td align="center" rowspan="2"><strong>UAV (avec tiges)</strong></td>
      <td align="center" rowspan="2">Classification</td>
      <td align="center">Couche de végétation</td>
      <td align="center">Régulière</td>
      <td align="center">12 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <tr>
      <td align="center">Tiges</td>
      <td align="center">Forêt mixte</td>
      <td align="center">8 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <!-- TLS Classification -->
    <tr>
      <td align="center" rowspan="7"><strong>TLS</strong></td>
      <td align="center" rowspan="7">Classification</td>
      <td align="center">Couche de végétation</td>
      <td align="center">Régulière</td>
      <td align="center">8 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <tr>
      <td align="center" rowspan="3">Tiges</td>
      <td align="center" rowspan="3">Boréale</td>
      <td align="center">10 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <tr>
      <td align="center">4 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <tr>
      <td align="center">20 cm</td>
      <td align="center">8 Go</td>
    </tr>
    <tr>
      <td align="center">Tiges</td>
      <td align="center">Régulière</td>
      <td align="center">4 cm</td>
      <td align="center">12 Go</td>
    </tr>
    <tr>
      <td align="center" rowspan="2">Tiges + branches</td>
      <td align="center" rowspan="2">Régulière</td>
      <td align="center">4 cm</td>
      <td align="center">2 Go</td>
    </tr>
    <tr>
      <td align="center">2,5 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <!-- ALS Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>ALS (ou UAV sans tiges)</strong></td>
      <td align="center" rowspan="2">Regroupement</td>
      <td align="center">Cimes des arbres</td>
      <td align="center">Site pétrolier</td>
      <td align="center">10 cm</td>
      <td align="center">4 Go</td>
    </tr>
    <tr>
      <td align="center">Segments d’arbres</td>
      <td align="center">Site pétrolier</td>
      <td align="center">10 cm</td>
      <td align="center">4 Go</td>
    </tr>
    <!-- UAV Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>UAV (avec tiges)</strong></td>
      <td align="center" rowspan="2">Regroupement</td>
      <td align="center">Bases d’arbres</td>
      <td align="center">Forêt mixte</td>
      <td align="center">10 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <tr>
      <td align="center">Segments d’arbres</td>
      <td align="center">Forêt mixte</td>
      <td align="center">15 cm</td>
      <td align="center">4 Go</td>
    </tr>
    <!-- TLS Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>TLS</strong></td>
      <td align="center" rowspan="2">Regroupement</td>
      <td align="center">Bases d’arbres</td>
      <td align="center">Boréale</td>
      <td align="center">10 cm</td>
      <td align="center">3 Go</td>
    </tr>
    <tr>
      <td align="center">Segments d’arbres</td>
      <td align="center">Boréale</td>
      <td align="center">15 cm</td>
      <td align="center">4 Go</td>
    </tr>
  </tbody>
</table>

## 🗂️ Structure des dossiers

```
TreeAIBox-main
│   TreeAIBox_Plugin_Installer.exe                  # Installateur Windows pour le plugin
│   CloudCompare_Python_Plugin.nsi                  # Configuration de l’installateur
│   treeaibox-header.bmp                            # Icône d’installation
│   treeaibox-welcome.bmp                           # Icône d’installation
│   dl_visualization.svg                            # Illustration de la structure DL principale
│   LICENSE.txt                                     # Fichier de licence
│   model_zoo.json                                  # Liste des modèles DL disponibles
│   README.md                                       # README
│   TODO.md                                         # Liste de tâches
│   TreeAIBox.py                                    # Programme Python principal
│   treeaibox_logo.ico                              # Logo du plugin
│   treeaibox_ui.html                               # Interface principale (vue web)
├───img                                             # Icônes et images de l’interface
└───modules                                         # Sous-modules de TreeAIBox
    ├───filter                                      # TreeFiltering et WoodCls
    │       componentFilter.py                      # Filtrage des couches arbres/branches/tiges
    │       createDTM.py                            # Création de DTM à partir des couches filtrées
    │       *.json                                  # Paramètres des modèles DL
    │       vox3DESegFormer.py                      # Structure du modèle DL (v2)
    │       vox3DSegFormer.py                       # Structure du modèle DL (v1)
    │       __init__.py
    │
    ├───qsm                                         # Module QSM
    │       applyQSM.py                             # Squelettisation et reconstruction 3D
    │       __init__.py
    │
    └───treeisonet
            cleanSmallerClusters.py                 # Suppression approximative des petits clusters
            treeLoc.py                              # Extraction des emplacements de tige/cime
            treeOff.py                              # Clustering de la couronne (scène sans tiges visibles)
            stemCluster.py                          # Clustering des tiges (règle du plus court chemin)
            crownCluster.py                         # Clustering couronne → tiges segmentées
            crownOff.py                             # Clustering deep learning de la couronne
            treeStat.py                             # Statistiques individuelles et par parcelle
            *.json                                  # Paramètres des modèles DL
            vox3DSegFormerDetection.py              # Détection de localisation d’arbres
            vox3DSegFormerRegression.py             # Régression de décalage de points
            __init__.py
```

## 🤝 Comment contribuer

1. Fork → branche de fonctionnalité → PR.
2. Respectez le style existant et ajoutez des tests si nécessaire.

Voir [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 Licence

Sauf indication contraire, le code source de ce projet est protégé par le droit d’auteur de la Couronne du gouvernement du Canada et est distribué sous la [Licence publique Creative Commons Attribution – Pas d’utilisation commerciale 4.0 International](LICENSE.txt).

Le mot-symbole Canada et les éléments graphiques associés à cette distribution sont protégés par la loi sur les marques de commerce et le droit d’auteur. Aucune permission n’est accordée pour les utiliser en dehors des paramètres du programme d’identité visuelle du gouvernement du Canada. Pour plus d’informations, voir [Exigences d’identité fédérale](https://www.canada.ca/en/treasury-board-secretariat/topics/government-communications/federal-identity-requirements.html).

Développé par Zhouxin Xi, testé par Charumitha Selvaraj

*Issu de plus d’une décennie de recherche LiDAR avec le soutien de collaborateurs dévoués.*

![capture d’écran finale](https://github.com/user-attachments/assets/2cac174d-f874-4a4a-bc4d-93c6ee9d4905)




