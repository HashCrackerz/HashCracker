# Progetto: Brute‑force su password hashed (CPU vs GPU)

## Obiettivo
Brute force su password "hashed" di lunghezza fissa; realizzare anche una versione CPU per confrontare gli effetti della computazione parallela su GPU.

### Estensioni (attività progettuale)
- Supporto per lunghezza password dinamica
- Attacco a dizionario
- Brute force su password hashed con *salt*
- Ottimizzazione gestione memoria e formattazione del README

---

## Struttura del progetto
```
progetto-password-crack/
├─ README.md
├─ src/
│  ├─ cpu/
│  │  └─ brute_cpu.c
│  ├─ gpu/
│  │  └─ brute_gpu.cu
│  └─ common/
│     └─ hash_utils.h
├─ data/
│  └─ wordlists/
└─ results/
```

---

## Scelte tecniche consigliate
- **Linguaggi:** C/C++ per CPU e CUDA (o OpenCL) per GPU. Python (con PyCUDA/PyOpenCL) se preferisci prototipare velocemente.
- **Algoritmi hash da testare:** SHA-256 (standard moderno), BLAKE2 (veloce), e per password con cost: Argon2/bcrypt (ma costoso da brute-force).
- **Salt:** supporta sia salt statico (per test) sia salt per-utente (tipico DB).

---

## Implementazione minima (requisiti d'esame)
1. **Generatore di password** per spazio di ricerca definito (alfabeto, lunghezza fissa L).
2. **Funzione hash** (usa libreria open-source o implementazione reference).
3. **Verifica**: confronta hash generato con hash target.
4. **Versione CPU** single/multi-threaded per baseline.
5. **Versione GPU** con kernel che mappa ogni thread a una o più candidate password.

---

## Estensioni dettagliate
- **Lunghezza dinamica:** genera combinazioni per lunghezze `min..max` e misura tempo totale per ogni lunghezza.
- **Attacco a dizionario:** scorrere wordlist; supportare trasformazioni (capitalization, leet, suffissi/prefissi numerici).
- **Salt:** memorizza salt e includilo nella funzione hash: `hash = H(salt || password)` o `H(password || salt)` a seconda del DB.

---

## Ottimizzazioni GPU (memoria & prestazioni)
- **Coalescing:** alloca buffer per le candidate in modo che letture globali siano coalesced.
- **Constant memory:** posiziona alfabeti e parametri fissi in `__constant__` (CUDA) per accesso veloce.
- **Shared memory:** usa per tabelle lookup o piccole strutture riutilizzate dai thread di un blocco.
- **Occupancy:** scegli number of threads per block per massimizzare occupancy; evita register spilling.
- **Riduzione dei branch:** implementa il generatore di password evitando branching pesante nel kernel.
- **Batching:** invia batch di candidate dalla CPU alla GPU per minimizzare overhead PCIe.
- **Gestione risultati:** usa meccanismo atomico o buffer di output con flag per scrivere la password trovata.

---

## Metriche e benchmark
- Tempo totale per spazio di ricerca
- Throughput: candidate/s
- Speedup: `T_cpu / T_gpu`
- Scalabilità rispetto a lunghezza e dimensione alfabetica

---

## Esempio di snippet (CPU - pseudocodice)
```c
for each candidate in keyspace:
    digest = sha256(candidate);
    if digest == target:
        print(candidate);
```

## Esempio di snippet (GPU - pseudocodice kernel)
```cuda
__global__ void try_candidates(...){
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    candidate = generate_from_index(idx, params);
    digest = sha256_gpu(candidate);
    if(digest == target){
        write_result(idx);
    }
}
```

---

## Problemi comuni con la visualizzazione di file Markdown e come risolverli
1. **Linee troppo lunghe / mancanza di newline:** assicurati di usare `\n` tra paragrafi e non concatenare tutto in una riga. I renderer (GitHub, VSCode) visualizzano peggio file con righe lunghe.
2. **Liste misformattate:** usa `- ` (trattino + spazio) o `1. ` per liste ordinate; evita caratteri speciali non ASCII in testata.
3. **Code fences:** racchiudi codice in triple backticks ``` per blocchi; specifica linguaggio (es. ```c) per syntax highlighting.
4. **Caratteri di tabulazione:** sostituisci tab con 2 o 4 spazi; alcuni renderer trattano male i tab.
5. **Encoding:** salva in UTF-8 senza BOM per garantire compatibilità.
6. **Immagini/risorse mancanti:** usa percorsi relativi corretti e assicurati che i file esistano.
7. **Uso di HTML inline:** alcuni renderer sanitizzano o non supportano alcune tag. Preferisci Markdown puro.

---

## Suggerimenti pratici per il tuo README.md (formattazione)
- Dividi in sezioni con `##` e `###` per leggibilità.
- Includi snippet di esempio con ` ``` `.
- Aggiungi una sezione "How to run" con comandi esatti per compilare e testare.
- Fornisci un piccolo `example_hashes.txt` con hash di prova e salt.

---

## "How to run" (esempio)
```
# Compila CPU
gcc -O3 src/cpu/brute_cpu.c -o brute_cpu -lcrypto

# Compila GPU (CUDA)
nvcc -O3 src/gpu/brute_gpu.cu -o brute_gpu

# Esegui
./brute_cpu --alphabet "abc..." --length 6 --target <hexhash>
./brute_gpu --alphabet ...
```

---

## Note finali
- Se vuoi, posso generare automaticamente i file `brute_cpu.c` e `brute_gpu.cu` con snippet funzionanti.
- Posso anche fornire script di benchmark e un `example_hashes.txt` per i test.

<!-- fine del documento -->
