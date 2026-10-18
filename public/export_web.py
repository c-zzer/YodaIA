"""Convertit modele.pt en fichiers lisibles par le navigateur : modele.json + poids.bin
Usage : python export_web.py   (dans le dossier mon-modele, après l'entraînement)
Les réglages de mini_gpt.py (BLOC, DIM, COUCHES, TETES) doivent être ceux utilisés pour l'entraînement."""
import json, torch
from mini_gpt import BLOC, DIM, COUCHES, TETES
ck = torch.load('modele.pt', map_location='cpu')
tensors, blobs, off = [], [], 0
for nom, t in ck['etat'].items():
    a = t.detach().cpu().float().contiguous().numpy()
    b = a.tobytes(); tensors.append({'name': nom, 'shape': list(a.shape), 'offset': off}); blobs.append(b); off += len(b)
json.dump({'bloc': BLOC, 'dim': DIM, 'couches': COUCHES, 'tetes': TETES, 'chars': ck['chars'], 'tensors': tensors},
          open('modele.json', 'w', encoding='utf-8'), ensure_ascii=False)
open('poids.bin', 'wb').write(b''.join(blobs))
print(f'modele.json et poids.bin créés ({off/1e6:.1f} Mo de poids).')
