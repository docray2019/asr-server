# Serveur de transcription MedTranscript (CPU)

faster-whisper (CTranslate2, 8 bits) derrière une file d'attente. Le navigateur envoie des blocs WAV
(1er bloc ≈ 8 s, puis ≈ 30 s coupés aux pauses) ; le serveur renvoie le texte de chaque bloc.
Le 1er bloc de chaque dictée passe devant les autres. Rien n'est écrit sur disque.

## Installation (Ubuntu, VM Infomaniak 4 vCPU / 4 Go)
```bash
sudo apt update && sudo apt install -y python3-venv ffmpeg
python3 -m venv ~/asr && source ~/asr/bin/activate
pip install -r requirements.txt
export ASR_TOKEN="$(openssl rand -hex 24)"; echo $ASR_TOKEN   # à noter
export ASR_MODEL=small ASR_WORKERS=1 ASR_CPU_THREADS=4
uvicorn app:app --host 0.0.0.0 --port 8000     # 1er lancement : télécharge le modèle (Hugging Face)
```
Ouvrir le port 8000 dans le pare-feu Infomaniak **uniquement pour votre IP** pendant l'essai.
Pour l'essai, pas de HTTPS ni de vraies données patient : utilisez un audio factice.

## Comparer Whisper et Parakeet
```bash
# Whisper medium (défaut)
export ASR_ENGINE=whisper ASR_MODEL=medium
# Parakeet TDT 0.6B v3 (25 langues dont le français, ~650 Mo en 8 bits)
export ASR_ENGINE=parakeet            # ASR_MODEL prend alors nemo-parakeet-tdt-0.6b-v3
```
Relancer `uvicorn`, puis le même `bench.py` sur le même enregistrement. Whisper complète chaque bloc à
30 s (≈ 5 s de calcul fixe sur 4 vCPU même pour 8 s de parole) ; Parakeet calcule proportionnellement
à la durée. Parakeet ignore le prompt (vocabulaire médical, contexte du bloc précédent) : comparer aussi
le texte obtenu, pas seulement la vitesse. `GET /health` indique le moteur.

## Réglages (variables d'environnement)
| Variable | Rôle | 4 vCPU / 4 Go |
|---|---|---|
| `ASR_ENGINE` | `whisper` ou `parakeet` | à comparer |
| `ASR_MODEL` | `small`, `medium`, `large-v3-turbo` | commencer par `small`, puis `medium` |
| `ASR_WORKERS` | transcriptions simultanées | 1 (2 si 8 vCPU) |
| `ASR_CPU_THREADS` | threads par transcription | 4 |
| `ASR_COMPUTE` | `int8` (défaut) | `int8` |
| `ASR_BEAM` | 1 = glouton (rapide) | 1 |

Mémoire : medium en 8 bits ≈ 1,5 à 2 Go par processus ; `small` ≈ 0,6 Go. Un seul processus ici
(les workers partagent le modèle).

## Essai de charge (depuis votre PC)
```bash
pip install httpx
python bench.py dictee.wav --url http://IP:8000 --token XXXX --sessions 1   # vitesse d'une session
python bench.py dictee.wav --url http://IP:8000 --token XXXX --sessions 4   # vos 3-4 sessions
python bench.py dictee.wav --url http://IP:8000 --token XXXX --sessions 4 --speed 10  # saturation rapide
```
`dictee.wav` : WAV 16 kHz mono 16 bits, de 2 à 5 minutes. Lire : « calcul » (secondes par bloc),
« file » (attente), « retard » (doit rester stable ; s'il grandit à chaque bloc, le serveur est saturé).
`GET /health` donne la vitesse mesurée (`speed_x_realtime`).

Tests de la file : `pytest test_queue.py`.

## Comparaison A/B sur les consultations de test (`compare_pipelines.py`)

A = MedTranscript avec le serveur GPU (Whisper large-v3) ; B = ce serveur CPU. Même découpage (blocs de 8 s,
recouvrement 1 s), puis la même chaîne : analyse Haiku → observation Sonnet 5.5 → courrier + résumé Haiku 5.5,
avec le profil de l'utilisateur `--as-user` (défaut drmaximelaporte@runbox.com). Chaque observation est comparée à
l'observation certifiée du dossier (similarité de texte, et note du juge Opus 4.8 avec `--judge`).

Prérequis : l'application déployée doit contenir la PR « as_user_email + judge-cr » (docray2019/medtranscript-next #217),
un compte **admin** (`MT_TOKEN` = jeton de session, ou `MT_EMAIL` + `MT_PASSWORD` sans 2FA), le serveur GPU allumé,
et le tunnel SSH vers ce serveur CPU ouvert.

```powershell
pip install httpx
$env:MT_EMAIL="vous@exemple.fr"; $env:MT_PASSWORD="..."        # ou  $env:MT_TOKEN="..."
python compare_pipelines.py "C:\Users\docra\OneDrive\Bureau\transcription audio\fichiers wav de tests\Epaule_MaxLaporte" --dry-run --only 1-3
python compare_pipelines.py "C:\...\Epaule_MaxLaporte" --only 1-3 --vps-url http://127.0.0.1:8000 --vps-token essai --judge
```
`--dry-run` vérifie les dossiers et fichiers trouvés sans aucun appel. Sortie : `resultats\<date>\rapport.html`
(côte à côte), `resume.csv` et, par dossier, les transcriptions, observations, courriers et `metrics.json`.
Chaque dossier coûte de vrais appels (Haiku ×4, Sonnet ×1, Haiku ×1 par pipeline, plus Opus si `--judge`).
`--resume` reprend un lot interrompu ; `--skip-a` / `--skip-b` ne lancent qu'un pipeline.
