# Modèle de l'estimateur de charge PHARE

Ce document décrit, de manière auto-suffisante, ce que calcule
`phare_load`, les hypothèses sous-jacentes, et les formules utilisées
pour chaque grandeur du rapport CLI. Il sert à la fois de référence
technique et de support pour la défense des chiffres lors d'un oral.

---

## 1. Objet

Estimer, pour une simulation magnétosphère globale avec **PHARE**:

- la **mémoire** (RAM particules) par niveau AMR;
- le **nombre de pas de temps** par niveau, sur une durée physique donnée;
- le **nombre de pushes** de particules par niveau;
- le **coût CPU·h** total;
- le **temps mural** en fonction du nombre de cœurs (scaling idéal).

Tous ces nombres sont produits pour:

- l'**hiérarchie AMR** (1 niveau MHD + plusieurs niveaux PIC raffinés);
- une ou plusieurs **références uniformes** (PIC plein-domaine sans
  raffinement).

L'outil est entièrement piloté par un fichier TOML
(`config.toml`); aucune modification de code n'est nécessaire pour
changer la résolution, la géométrie, le vent solaire, le nombre de
niveaux, etc.

---

## 2. Conventions

### 2.1 Système de coordonnées

Standard GSE (Geocentric Solar Ecliptic):

- `+x` pointe vers le Soleil
- `-x` pointe vers la queue magnétique
- Distances en rayons terrestres (Re) avec `Re = 6371.2 km`

Les modèles de Shue (magnétopause) et Jélínek (choc d'étrave) sont
évalués dans la même convention; aucun retournement n'est appliqué
en interne. Le tracé 2D inverse l'axe x à la fin pour des raisons de
lisibilité (Soleil à gauche, convention magnétosphérique).

### 2.2 Hiérarchie AMR

- Les niveaux sont listés du **plus grossier au plus fin** dans `[[levels]]`.
- Le premier niveau (`L0`) peut être MHD (sans particules) ou PIC.
- Les niveaux PIC suivants sont **emboîtés** par construction:
  $\text{masque}(L_{n+1}) \subset \text{masque}(L_n)$.
- Ratio de raffinement spatial supposé constant: $\Delta x_{n+1} = \Delta x_n / 2$.
- Ratio de raffinement temporel: $\Delta t_{n+1} = \Delta t_n / r$
  avec `r = dt_ratio_per_level` (défaut 4, soit $\Delta t \propto \Delta x^2$).

---

## 3. Modèles physiques

### 3.1 Magnétopause de Shue 1998

$$
r_{MP}(\theta) = r_0 \left(\frac{2}{1+\cos\theta}\right)^{\alpha}
$$

avec

$$
r_0 = (10.22 + 1.29 \tanh[0.184(B_z + 8.14)]) \, P_d^{-1/6.6}
$$

$$
\alpha = (0.58 - 0.007\,B_z)(1 + 0.024 \ln P_d)
$$

- $\theta$: angle depuis l'axe Terre–Soleil;
- $P_d$ en nPa (pression dynamique du vent solaire);
- $B_z$ en nT (composante nord-sud du IMF).

### 3.2 Choc d'étrave de Jélínek 2012

$$
r_{BS}(\theta) = R \left(\frac{2}{1+\cos\theta}\right)^{\lambda}
\quad \text{avec} \quad
R = 15.02 \, P_d^{-1/6.55}, \quad \lambda = 1.17
$$

### 3.3 Domaine de validité

Ces fits divergent quand $\theta \to \pi$ (queue lointaine). Le code
plafonne $1+\cos\theta$ à $10^{-3}$ pour la sécurité numérique, et
l'option `dayside_only = true` (défaut) restreint les niveaux PIC à
la moitié `+x ≥ 0` du domaine, où les fits sont fiables.

### 3.4 Pression dynamique

$$
P_d \;[\mathrm{nPa}] = 1.6726 \times 10^{-6} \times n\,[\mathrm{cm^{-3}}] \times V^2\,[\mathrm{km/s}]^2
$$

### 3.5 Bouton "dipole_strength"

Permet de simuler une magnétosphère plus petite ou plus grande qu'à
champ terrestre nominal. À l'équilibre de pression subsolaire,
$r_{MP} \propto M_E^{1/3}$ → on multiplie $r_{MP}$ et $r_{BS}$ par
$(\text{dipole\_strength})^{1/3}$.

- `dipole_strength = 1.0` → Terre actuelle (MP subsolaire ≈ 10 Re)
- `dipole_strength = 0.5` → magnétosphère ~21% plus petite
- `dipole_strength = 0.125` → moitié de taille
- `dipole_strength = 0.001` → MP à ~1 Re

Cette mise à l'échelle est **exacte** au point subsolaire et
**approchée** ailleurs (l'aplatissement $\alpha$ et $\lambda$ ne sont
pas reparamétrés).

---

## 4. Géométrie des régions AMR

### 4.1 Sampling

Le domaine est échantillonné sur un lattice cubique uniforme avec un
pas `sample_dx_re` (défaut 0.5 Re). Pour chaque cellule du probe lattice
on calcule:

- la distance géocentrique $r = \sqrt{x^2+y^2+z^2}$,
- l'angle $\theta = \arccos(x/r)$,
- les rayons $r_{MP}(\theta)$ et $r_{BS}(\theta)$ aux valeurs du vent solaire.

### 4.2 Types de régions

Chaque niveau PIC choisit sa région via `region = "..."`:

| `region`  | Définition mathématique                                                 | Paramètre   |
|-----------|--------------------------------------------------------------------------|-------------|
| `"full"`  | Tout le domaine                                                           | —           |
| `"shell"` | $r_{MP}(\theta) - \mathrm{pad} \le r \le r_{BS}(\theta) + \mathrm{pad}$  | `pad_re`    |
| `"band"`  | $|r - r_X(\theta)| \le \mathrm{band}$ pour $X \in \mathrm{boundaries}$    | `band_re`, `boundaries` |

`boundaries` est une liste contenant `"mp"`, `"bs"`, ou les deux.

### 4.3 Restrictions

Après calcul du masque brut:

1. Si `dayside_only = true`: intersection avec $\{x \ge 0\}$.
2. **Emboîtement**: intersection avec le masque du niveau PIC
   précédent (plus grossier), garantissant $L_{n+1} \subset L_n$.

### 4.4 Volume

$$
V_{n} \,[\mathrm{Re}^3] = (\text{nb cellules masquées}) \times (\text{sample\_dx\_re})^3
$$

C'est une approximation de Riemann; sa précision augmente quand
`sample_dx_re` diminue (au prix du temps de calcul et de la RAM du
script).

---

## 5. Modèle de coût mémoire

Pour un niveau PIC de résolution $\Delta x$ couvrant un volume $V$:

$$
N_\text{cells} = V \,[\mathrm{km}^3] / (\Delta x)^3
$$

$$
N_\text{part} = \mathrm{PPC} \times N_\text{cells}
$$

$$
\mathrm{RAM} = N_\text{part} \times \mathrm{bytes\_per\_particle}
$$

avec:

- `PPC = 100` (particles per cell, défaut);
- `bytes_per_particle = 76 B` (valeur mesurée dans PHARE: position 3×8 B,
  vitesse 3×8 B, poids 8 B, charge/AMR overhead ~20 B).

Les niveaux MHD (`kind = "mhd"`) sont comptés pour 0 particule, 0 RAM
(seul le maillage MHD vit sur le niveau, et son coût n'est pas inclus
dans cet estimateur — voir §10 limitations).

---

## 6. Modèle de coût temporel

### 6.1 Pas de temps des niveaux AMR

PHARE/SAMRAI utilise du **subcycling**: chaque niveau plus grossier
avance avec un pas $\Delta t$ plus grand que le niveau suivant, selon
un ratio temporel imposé par le ratio spatial.

Soit `r = dt_ratio_per_level` (défaut 4). Pour $N$ niveaux indexés
0..N−1 du plus grossier au plus fin:

$$
\Delta t_n = \Delta t_\text{finest} \times r^{(N-1-n)}
$$

Le pas du niveau le plus fin est dérivé de l'ancrage L1 (premier
niveau PIC):

$$
\Delta t_\text{finest} = \Delta t_{L_1} \times \left(\frac{\Delta x_\text{finest}}{\Delta x_{L_1}}\right)^2
$$

avec `dt_L1_omega_ci = 0.05` (en unités de $\Omega_{ci}^{-1}$) par défaut
et `omega_ci_inverse_s = 1` s. Ceci suppose une **CFL en $\Delta x^2$**,
appropriée pour une advection de particules à vitesse constante limitée
par la cellule (en pratique limite Whistler dominante pour PIC dans la
magnétosphère; la formule est conservatrice).

Optionnellement, `dt_finest_s` peut être fixé en absolu et remplace
cette dérivation.

### 6.2 Nombre de pas par niveau

Sur la durée totale `target_hours`:

$$
N_\text{finest} = \frac{T_\text{run}}{\Delta t_\text{finest}}
$$

Pour un niveau $n$:

$$
N_\text{steps}(n) = N_\text{finest} \times \text{steps\_per\_finest}(n)
$$

où `steps_per_finest(n)` $= r^{-(N-1-n)}$ représente la fraction de pas
du niveau le plus fin pendant lesquels le niveau $n$ s'avance.

### 6.3 Pas de temps de la référence uniforme

Une **vraie** simulation uniforme à $\Delta x_\text{ref}$ utiliserait
son propre CFL: $\Delta t \propto \Delta x^2$. L'estimateur applique
ce scaling à chaque référence:

$$
\Delta t_\text{ref} = \Delta t_\text{finest} \times \left(\frac{\Delta x_\text{ref}}{\Delta x_\text{finest}}\right)^2
$$

$$
N_\text{steps,ref} = N_\text{finest} \times \left(\frac{\Delta x_\text{finest}}{\Delta x_\text{ref}}\right)^2
$$

Conséquences:

- Si $\Delta x_\text{ref} = \Delta x_\text{finest}$ (réf. fine 10 km par
  défaut): $N_\text{steps,ref} = N_\text{finest}$ — la référence partage
  exactement le pas le plus fin de l'AMR.
- Si $\Delta x_\text{ref} = 10 \Delta x_\text{finest}$ (réf. coarse 100
  km par défaut): $N_\text{steps,ref}$ est 100 fois plus petit.

C'est ce qui rend la comparaison "AMR vs coarse uniforme" physiquement
honnête: chaque run utilise le $\Delta t$ adapté à sa propre
résolution.

---

## 7. Modèle de coût CPU

### 7.1 Pushes par niveau

$$
\text{pushes}(n) = N_\text{steps}(n) \times N_\text{part}(n)
$$

Pour l'hiérarchie AMR:

$$
\text{pushes}_\text{AMR} = \sum_{n \in \text{PIC}} \text{pushes}(n)
$$

### 7.2 Coût single-thread

$$
\text{CPU·s} = \text{pushes} \times \text{sec\_per\_particle\_per\_step}
$$

avec `sec_per_particle_per_step = 10 ns` par défaut — valeur typique
pour un solveur PIC explicite optimisé sur CPU moderne.

### 7.3 Wall-time à N cœurs

$$
T_\text{wall}(N_\text{cores}) = \frac{\text{CPU·s}}{N_\text{cores}}
$$

**Hypothèse: scaling parfaitement linéaire.** En pratique, à >$10^5$
cœurs, l'efficacité réelle est plutôt 50–70%; les nombres en bas du
rapport CLI sont donc **optimistes** d'un facteur ~1.5–2 à grande
échelle.

---

## 8. Référence(s) uniforme(s)

`reference_dx_di` (ou `reference_dx_km`) peut être:

- un **scalaire** → une seule référence;
- une **liste** → plusieurs références comparées simultanément.

Le défaut `[1.0, 0.1]` produit:

- **Uniform coarse 100 km = 1 δᵢ**: PIC sans aucun raffinement, à la
  résolution d'un MHD-cinétique grossier. Représente un baseline
  "pas cher mais ne résout rien de cinétique".
- **Uniform fine 10 km = 0.1 δᵢ**: PIC qui résout les échelles ioniques
  partout dans le domaine. Représente la solution complète qu'il faudrait
  faire sans AMR — invariablement hors de portée.

Le rapport AMR / coarse / fine encadre la valeur ajoutée de l'AMR:

- AMR **coûte plus** que la coarse, mais **résout** la physique cinétique
  là où elle est essentielle (magnétopause).
- AMR **coûte beaucoup moins** que la fine, sans sacrifier la
  physique aux endroits critiques.

---

## 9. Sortie du CLI — section par section

### Section 1 — Memory footprint

Pour chaque niveau (incluant MHD): `N_cells`, `N_part`, `RAM`.
Total AMR + RAM de chaque référence uniforme.

### Section 2 — Timesteps over the run

Pour chaque niveau AMR: $\Delta t$ et `N_steps` sur la durée totale.
Pour chaque référence: $\Delta t_\text{ref}$ propre et `N_steps_ref`.

### Section 3 — Particle pushes per level

Pour chaque niveau PIC: pushes total. Somme AMR + pushes de chaque
référence + ratio (AMR/uniform ou uniform/AMR selon le signe).

### Section 4 — Work breakdown per finest-level step

Décomposition du coût par pas du niveau le plus fin:

$$
w_n = \text{steps\_per\_finest}(n) \times N_\text{part}(n)
$$

Indique quel niveau domine. Dans la config par défaut, **L4 fait
93% du travail par pas** — par construction, le niveau le plus fin
porte la majorité du coût car il a le plus de particules ET avance
le plus souvent.

### Section 5 — CPU·hours

Total AMR et total de chaque référence, plus le ratio. Le verdict
"AMR cheaper" / "uniform cheaper" est calculé pour chaque référence.
Tableau wall-time à 1 / 10⁴ / 10⁵ / 10⁶ cœurs.

---

## 10. Limites et hypothèses non triviales

| Sujet | Hypothèse | Effet potentiel |
|---|---|---|
| Coût MHD | Niveaux MHD comptés pour 0 | L0 a en réalité un coût non nul (champ E/B avancé); marginal vs PIC |
| Patches SAMRAI | Volume-équivalent, pas patch-équivalent | Overhead réel patches: +20-50% sur RAM/cells au plus grossier |
| CFL | $\Delta t \propto \Delta x^2$ | Conservatrice; CFL Whistler peut imposer un pas plus petit dans certains régimes |
| Cost-per-push | 10 ns constant | Ignore overhead halo/MPI, cache misses sur très grands runs |
| Scaling cœurs | Linéaire idéal | À >10⁵ cœurs, efficacité 50-70%; sous-estime wall-time |
| Fits Shue/Jélínek | Dayside-only par défaut | Au-delà de θ≈130° les fits divergent — d'où la restriction |
| dipole_strength | Mise à l'échelle subsolaire seule | $\alpha$, $\lambda$ inchangés → forme approximative loin du nez |
| Intégration de volume | Riemann sur lattice 0.5 Re | Erreur ~5-10% sur V des coquilles fines (à raffiner pour publication) |
| PPC fixe | 100 partout | En pratique on peut sous-peupler le tail; économie potentielle non modélisée |

---

## 11. Vérification analytique (config par défaut)

Constantes:

- $V_\text{dom} = 150 \times 100 \times 100 = 1.5 \times 10^6 \mathrm{Re}^3$
- $\mathrm{Re}^3 = 2.585 \times 10^{11} \mathrm{km}^3$
- $V_\text{dom} = 3.878 \times 10^{17} \mathrm{km}^3$
- $T_\text{run} = 3600$ s

### 11.1 Niveau le plus fin (L4, $\Delta x = 10$ km)

$$
\Delta t_\text{L4} = 0.05 \times (10/80)^2 = 0.05/64 = 7.8125 \times 10^{-4} \, \mathrm{s}
$$

$$
N_\text{finest} = 3600 / 7.8125 \times 10^{-4} = 4\,608\,000
$$

Volume L4 (band ±0.25 Re autour MP, dayside, dans L3) ≈ **553 Re³**
(intégration lattice).

$$
N_\text{cells} = 553 \times (6371.2/10)^3 = 553 \times 2.585 \times 10^8 \approx 1.430 \times 10^{11}
$$

$$
N_\text{part} = 100 \times N_\text{cells} = 1.430 \times 10^{13}
$$

$$
\text{pushes L4} = 4.608 \times 10^6 \times 1.430 \times 10^{13} \approx 6.59 \times 10^{19}
$$

### 11.2 Total AMR

Σ pushes des 4 niveaux PIC ≈ $7.06 \times 10^{19}$.

$$
\text{CPU·s} = 7.06 \times 10^{19} \times 10^{-8} = 7.06 \times 10^{11} \, \mathrm{s}
$$

$$
\text{CPU·h} \approx 1.96 \times 10^8 = 196 \, \mathrm{M\,CPU \cdot h}
$$

### 11.3 Référence coarse 100 km

$$
N_\text{cells} = 3.878 \times 10^{17} / 100^3 = 3.878 \times 10^{11}
$$

$$
N_\text{part} = 3.878 \times 10^{13}
\quad\Rightarrow\quad
\mathrm{RAM} = 76 \times N_\text{part} = 2.95 \times 10^{15} \, \mathrm{B} = 2.62 \, \mathrm{PB}
$$

$$
\Delta t_\text{ref} = 7.8125 \times 10^{-4} \times (100/10)^2 = 7.8125 \times 10^{-2} \, \mathrm{s}
$$

$$
N_\text{steps,ref} = 46\,080
\quad\Rightarrow\quad
\text{pushes} = 1.79 \times 10^{18}
\quad\Rightarrow\quad
\text{CPU·h} \approx 4.97 \, \mathrm{M}
$$

### 11.4 Référence fine 10 km

Échelles relatives par rapport à la coarse:

- $N_\text{cells}$: $\times 10^3$
- $N_\text{part}$: $\times 10^3$ → $3.878 \times 10^{16}$
- RAM: $\times 10^3$ → $2.56 \mathrm{EB}$
- $\Delta t$: $\times 10^{-2}$ → identique à L4
- $N_\text{steps}$: $\times 10^2$ → $4.6 \times 10^6$
- pushes: $\times 10^5$ → $1.79 \times 10^{23}$
- CPU·h: $\times 10^5$ → **$497 \times 10^9$** ≈ 497 milliards.

### 11.5 Ratios résumés

$$
\frac{\text{AMR}}{\text{coarse}} \approx 39.5 \, (\text{AMR plus cher: il fait de la cinétique})
$$

$$
\frac{\text{fine}}{\text{AMR}} \approx 2531 \, (\text{AMR évite ~2500× le coût d'une fine partout})
$$

$$
\frac{\text{fine}}{\text{coarse}} = 10^5 \quad (\Delta x^{-5} : 10^{-3} \text{ cells} \times 10^{-2} \text{ dt})
$$

---

## 12. Comment reproduire

```bash
.venv/bin/python -m phare_load.cli --config config.toml
```

Pour explorer la sensibilité:

- Multi-références: éditer `reference_dx_di = [1.0, 0.5, 0.1]` dans
  `config.toml` pour ajouter une référence intermédiaire.
- Magnétosphère plus petite: `dipole_strength = 0.1` (volumes des
  niveaux fortement réduits).
- Raffinement supplémentaire: ajouter `[[levels]]` avec `dx_di = 0.05`.

Voir [README.md](README.md) pour l'utilisation complète et la visualisation 3D.
