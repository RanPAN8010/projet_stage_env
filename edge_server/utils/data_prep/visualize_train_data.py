import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Génère et sauvegarde trois graphiques (nuage de points, boîtes à moustaches et 
# courbes de densité) pour analyser les profils environnementaux statiques et dynamiques.
def generate_environmental_plots():
    current_path = os.path.abspath(__file__)
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        print("Erreur : Le script n'est pas placé dans le dossier 'edge_server' !")
        return
        
    data_path = os.path.join(base_project_dir, 'data', 'final_train_data_5_features.csv')
    output_dir = os.path.join(base_project_dir, 'data')
    
    if not os.path.exists(data_path):
        print(f"Erreur : Fichier introuvable - {data_path}")
        return
        
    print("Chargement du jeu de données d'entraînement...")
    df = pd.read_csv(data_path)
    
    # 建立显式的标签映射，以便生成清晰的图例
    # Établir un mappage explicite des étiquettes afin de générer une légende claire
    mapping_labels_francais = {
        0: "Sécurité",
        1: "Canicule",
        2: "Feu / Fumée"
    }
    df['État Environnemental'] = df['Label'].map(mapping_labels_francais)
    
    # Configuration du thème graphique / 配置图表主题
    sns.set_theme(style="whitegrid")
    palette_couleurs = {
        "Sécurité": "#3498db",      # Bleu pour Sécurité
        "Canicule": "#e67e22",      # Orange pour Canicule
        "Feu / Fumée": "#e74c3c"    # Rouge pour Feu / Fumée
    }
    
    # 计算分位数，以便在可视化时自动剔除极端离群点（Outliers）造成的画面拉伸
    # Calculer les quantiles pour éliminer automatiquement les valeurs aberrantes (outliers) et éviter la déformation des graphiques
    q_low_temp, q_hi_temp = df['Temperature'].quantile(0.005), df['Temperature'].quantile(0.995)
    q_low_hum, q_hi_hum = df['Humidity'].quantile(0.005), df['Humidity'].quantile(0.995)
    q_low_tr, q_hi_tr = df['Temp_Rate'].quantile(0.01), df['Temp_Rate'].quantile(0.99)
    q_low_hr, q_hi_hr = df['Humidity_Rate'].quantile(0.01), df['Humidity_Rate'].quantile(0.99)


    # 图表 1 : 散点图 (温度 vs 湿度的绝对空间分布)
    # Graphique 1 : Nuage de points 
    print("Génération du Graphe 1 : Nuage de points...")
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=df,
        x='Temperature',
        y='Humidity',
        hue='État Environnemental',
        palette=palette_couleurs,
        alpha=0.4,
        s=15,
        edgecolor='none'
    )
    
    plt.xlim(q_low_temp - 2, q_hi_temp + 2)
    plt.ylim(q_low_hum - 5, q_hi_hum + 5)
    
    plt.title("Cartographie de l'Espace Environnemental (Température vs Humidité)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Température (°C)", fontsize=11, labelpad=8)
    plt.ylabel("Humidité (%)", fontsize=11, labelpad=8)
    plt.legend(title="Diagnostic AI", loc='lower left')
    plt.tight_layout()
    
    img_path_1 = os.path.join(output_dir, 'visualisation_espace_environnemental.png')
    plt.savefig(img_path_1, dpi=300)
    plt.close()
    
    # 图表 2 : 箱线图 (静态特征与动态变动率的分布对比)
    # Graphique 2 : Boîtes à moustaches 
    print("Génération du Graphe 2 : Boîtes à moustaches...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 温度绝对值分布箱线图
    # Boîte à moustaches de la distribution de la température absolue
    sns.boxplot(ax=axes[0, 0], data=df, x='État Environnemental', y='Temperature', hue='État Environnemental', palette=palette_couleurs, legend=False, fliersize=1)
    axes[0, 0].set_title("Distribution de la Température Absolue", fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel("Température (°C)")
    axes[0, 0].set_xlabel("")
    
    # 湿度绝对值分布箱线图
    # Boîte à moustaches de la distribution de l'humidité absolue
    sns.boxplot(ax=axes[0, 1], data=df, x='État Environnemental', y='Humidity', hue='État Environnemental', palette=palette_couleurs, legend=False, fliersize=1)
    axes[0, 1].set_title("Distribution de l'Humidité Absolue", fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel("Humidité (%)")
    axes[0, 1].set_xlabel("")
    
    # 温度变化率分布箱线图
    # Boîte à moustaches du taux de variation de la température
    sns.boxplot(ax=axes[1, 0], data=df, x='État Environnemental', y='Temp_Rate', hue='État Environnemental', palette=palette_couleurs, legend=False, fliersize=1)
    axes[1, 0].set_ylim(q_low_tr - 0.5, q_hi_tr + 0.5)
    axes[1, 0].set_title("Taux de Variation de la Température", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("Variation (°C/unité)")
    axes[1, 0].set_xlabel("État Environnemental")
    
    # 湿度变化率分布箱线图
    # Boîte à moustaches du taux de variation de l'humidité
    sns.boxplot(ax=axes[1, 1], data=df, x='État Environnemental', y='Humidity_Rate', hue='État Environnemental', palette=palette_couleurs, legend=False, fliersize=1)
    axes[1, 1].set_ylim(q_low_hr - 1.0, q_hi_hr + 1.0)
    axes[1, 1].set_title("Taux de Variation de l'Humidité", fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel("Variation (%/unité)")
    axes[1, 1].set_xlabel("État Environnemental")
    
    plt.suptitle("Analyse Comparative des Profils Statiques et Dynamiques par État", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path_2 = os.path.join(output_dir, 'comparaison_profils_environnementaux.png')
    plt.savefig(img_path_2, dpi=300)
    plt.close()
    

    # 图表 3 : 直方图与密度曲线 (分析动态跳变 Delta Net 的数学可分性)
    # Graphique 3 : Histogrammes et courbes de densité
    print("Génération du Graphe 3 : Courbes de densité des taux de variation...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 温度变化量的概率密度分布
    # Distribution de densité de probabilité de la variation de température
    sns.histplot(
        ax=axes[0], data=df, x='Temp_Rate', hue='État Environnemental',
        palette=palette_couleurs, element='step', stat='density', common_norm=False, alpha=0.25, kde=True
    )
    axes[0].set_xlim(q_low_tr - 0.2, q_hi_tr + 0.2)
    axes[0].set_title("Distribution de Probabilité : Variation Thermique Dynamique", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Delta Température (Différence de valeur entre deux pas)")
    axes[0].set_ylabel("Densité")
    
    # 湿度变化量的概率密度分布
    # Distribution de densité de probabilité de la variation d'humidité
    sns.histplot(
        ax=axes[1], data=df, x='Humidity_Rate', hue='État Environnemental',
        palette=palette_couleurs, element='step', stat='density', common_norm=False, alpha=0.25, kde=True
    )
    axes[1].set_xlim(q_low_hr - 0.5, q_hi_hr + 0.5)
    axes[1].set_title("Distribution de Probabilité : Variation Hydrométrique Dynamique", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Delta Humidité (Différence de valeur entre deux pas)")
    axes[1].set_ylabel("Densité")
    
    plt.suptitle("Analyse Intégrale des Courbes de Densité Dynamiques (Vitesse de Transition)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path_3 = os.path.join(output_dir, 'analyse_densite_variations_deltas.png')
    plt.savefig(img_path_3, dpi=300)
    plt.close()
    
    print("\n" + "="*60)
    print("Visualisation du Dataset d'Entraînement Terminée avec Succès !")
    print("="*60)
    print(f"1. Scatter plot - Température/Humidité : {img_path_1}")
    print(f"2. Boxplot - Statistiques par diagnostic : {img_path_2}")
    print(f"3. Courbe de densité - Vitesse de saut : {img_path_3}")
    print("="*60 + "\n")

if __name__ == "__main__":
    generate_environmental_plots()