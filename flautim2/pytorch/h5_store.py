"""
h5_store.py
==================

Arquivo único (módulo) para salvar:
- Eventos (logs, measures, etc.) em HDF5 como "tabela" (JSON por linha)
- Outputs (imagens, arquivos, texto/json, matrizes) em HDF5, retornando um "ref"
- Merge: juntar vários shards (um por processo) em um arquivo HDF5 final

POR QUE ISSO FUNCIONA COM THREADS / PROCESSOS / RAY?
----------------------------------------------------
- Ray normalmente cria vários processos (workers).
- HDF5 (via h5py) não é seguro para múltiplos escritores no MESMO arquivo.
- Solução simples e robusta: cada PROCESSO escreve no seu próprio arquivo shard:
    output/shard_<experiment_id>_pid<PID>.h5

Dentro do mesmo processo, pode haver threads:
- Para evitar duas threads escreverem ao mesmo tempo no mesmo shard do processo,
  usamos um threading.Lock (_process_lock).

ARQUIVOS GERADOS (todos no mesmo diretório base_dir):
----------------------------------------------------
- Shards:
    base_dir/shard_<experiment_id>_pid12345.h5
    base_dir/shard_<experiment_id>_pid12346.h5
    ...

- Arquivo final (merge):
    base_dir/merged_<experiment_id>.h5

ESTRUTURA DENTRO DO HDF5:
-------------------------
1) Eventos (qualquer "collection"):
    /<collection>/events                -> dataset append-only (cada linha = JSON UTF-8 em bytes)

2) Outputs:
    /outputs/blobs/<ref>                -> bytes (imagem/arquivo/texto/json)
    /outputs/arrays/<ref>               -> dataset numpy (matriz/ndarray)
    /outputs/meta/<ref>                 -> metadados para blobs (JSON em bytes)
    /outputs/meta/<ref>__arraymeta       -> metadados para arrays (JSON em bytes)

NOTA:
-----
- Para outputs "pesados", o ideal é salvar no /outputs/... e colocar apenas o "ref" no evento.
- Isso mantém eventos leves e torna tudo mais rápido/organizado.

Dependências:
-------------
pip install h5py numpy
"""

from __future__ import annotations

import os
import json
import time
import glob
import hashlib
import threading
from typing import Any, Dict, Optional, Iterable, List, Set
from pathvalidate import sanitize_filename

import h5py
import numpy as np
 

# ============================================================================
# Concurrency: lock apenas entre THREADS do mesmo processo.
# (Entre PROCESSOS não há lock porque cada processo escreve no seu próprio shard)
# ============================================================================
_process_lock = threading.Lock()


# ============================================================================
# Helpers básicos (nomes/paths)
# ============================================================================
def shard_path(base_dir: str, experiment_id: str) -> str:
    """
    Caminho do shard do processo atual.

    Exemplo:
      output/shard_exp42_pid12345.h5

    - experiment_id identifica o experimento
    - pid identifica o processo (Ray worker, multiprocess, etc.)
    """
    os.makedirs(base_dir, exist_ok=True)
    pid = os.getpid()
    experiment_id = sanitize_filename(experiment_id)
    return os.path.join(base_dir, f"shard_{experiment_id}_pid{pid}.h5")


def list_shards(base_dir: str, experiment_id: str) -> List[str]:
    """
    Lista todos os shards daquele experimento no diretório base_dir.
    """
    experiment_id = sanitize_filename(experiment_id)
    pattern = os.path.join(base_dir, f"shard_{experiment_id}_pid*.h5")
    return sorted(glob.glob(pattern))


def default_merged_path(base_dir: str, experiment_id: str) -> str:
    """
    Caminho padrão do arquivo final mesclado.
    """
    experiment_id = sanitize_filename(experiment_id)
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"merged_{experiment_id}.h5")


def _sha256(data: bytes) -> str:
    """
    Identificador único para blobs (dedup simples).
    """
    return hashlib.sha256(data).hexdigest()


def _sanitize_for_json(x):
    """
    Converte um objeto Python arbitrário em algo que seja
    100% serializável em JSON.
 
    -------------------
    O MongoDB (BSON) aceita tipos que JSON NÃO aceita.
    Já o nosso armazenamento em HDF5 grava eventos como JSON UTF-8.

    Exemplos de tipos comuns que aparecem nos logs e quebram JSON:
      - bson.ObjectId            -> TypeError ao fazer json.dumps
      - numpy.float32 / int64    -> TypeError ao fazer json.dumps
      - numpy.ndarray            -> TypeError (array não é JSON)
      - bytes                    -> TypeError
      - datetime, Path, Exception, Tensor, etc.

    Se NÃO converter esses tipos antes:
      - o logger pode quebrar no meio do experimento
      - o worker (Ray/multiprocess) pode morrer
      - o arquivo HDF5 pode ficar inconsistente
      - você perde logs/experimentos inteiros

    Esta função garante que:
      - tudo que chega aqui vira JSON válido
      - logs NUNCA derrubam o experimento
    """

    # ------------------------------------------------------------
    # 1) ObjectId (MongoDB)
    # ------------------------------------------------------------
    # MongoDB usa ObjectId como chave primária (_id).
    # JSON não sabe serializar ObjectId.
    #
    # Exemplo real:
    #   {"_id": ObjectId("65f1a...")}
    #
    # Sem sanitização:
    #   TypeError: ObjectId is not JSON serializable
    #
    # Com sanitização:
    #   {"_id": "65f1a..."}
    try:
        from bson import ObjectId
        if isinstance(x, ObjectId):
            return str(x)
    except Exception:
        # bson pode não estar instalado (ex: sem Mongo)
        pass

    # ------------------------------------------------------------
    # 2) Tipos escalares do NumPy
    # ------------------------------------------------------------
    # numpy.float32, numpy.int64, etc. NÃO são JSON nativos.
    #
    # Exemplo real:
    #   {"value": np.float32(0.91)}
    #
    # Sem sanitização:
    #   TypeError: Object of type float32 is not JSON serializable
    #
    # Com sanitização:
    #   {"value": 0.91}
    try:
        import numpy as _np
        if isinstance(x, (_np.integer, _np.floating)):
            return x.item()
    except Exception:
        pass

    # ------------------------------------------------------------
    # 3) Arrays NumPy
    # ------------------------------------------------------------
    # Arrays não são JSON.
    #
    # Estratégia aqui:
    #   - converter para lista (tolist)
    #
    # OBS:
    #   Isso é adequado para arrays PEQUENOS (ex: métricas).
    #   Para arrays GRANDES (pesos, embeddings),
    #   o ideal é usar save_output(kind="array")
    #   e guardar só o ref no evento.
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass

    # ------------------------------------------------------------
    # 4) Dicionários
    # ------------------------------------------------------------
    # Percorre recursivamente e sanitiza chaves/valores.
    #
    # Exemplo:
    #   {"config": {"lr": np.float32(1e-3)}}
    #
    # Resultado:
    #   {"config": {"lr": 0.001}}
    if isinstance(x, dict):
        return {str(k): _sanitize_for_json(v) for k, v in x.items()}

    # ------------------------------------------------------------
    # 5) Listas e tuplas
    # ------------------------------------------------------------
    # Sanitiza cada elemento recursivamente.
    #
    # Exemplo:
    #   [1, np.float32(2.0), ObjectId(...)]
    #
    # Resultado:
    #   [1, 2.0, "65f1a..."]
    if isinstance(x, (list, tuple)):
        return [_sanitize_for_json(v) for v in x]

    # ------------------------------------------------------------
    # 6) Bytes / bytearray
    # ------------------------------------------------------------
    # JSON não aceita bytes.
    #
    # Estratégia:
    #   - tentar decodificar como UTF-8
    #   - se falhar, usar repr() para não perder informação
    #
    # Exemplo:
    #   b"hello" -> "hello"
    #   b"\xff\x00" -> "b'\\xff\\x00'"
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return repr(x)
 
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


# ============================================================================
# Eventos (logs / measures / experiments / etc.)
# ============================================================================
def _ensure_events_dataset(h5: h5py.File, key: str) -> h5py.Dataset:
    """
    Garante que exista um dataset append-only de eventos no caminho `key`.

    Por que vlen uint8?
    - Cada evento é um JSON, e JSON tem tamanho variável.
    - Guardamos como bytes UTF-8.
    """
    if key in h5:
        return h5[key]

    vlen = h5py.vlen_dtype(np.dtype("uint8"))
    return h5.create_dataset(
        key,
        shape=(0,),          # começa vazio
        maxshape=(None,),    # cresce indefinidamente
        dtype=vlen,
        chunks=(1024,),      # chunk para desempenho no append
    )


def save_event(
    *,
    base_dir: str,
    experiment_id: str,
    collection: str,
    doc: Dict[str, Any],
) -> None:
    """
    Salva UM evento na coleção `collection`.

    Exemplos:
      collection="logs"
      collection="measures"
      collection="experiments"
      collection="experiment_results"

    O evento vai para:
      /<collection>/events

    Conteúdo salvo por linha:
      {"ts": time.time(), "experiment_id": <experiment_id>, **doc}
    """
    path = shard_path(base_dir, experiment_id)
    key = f"/{collection}/events"

    # garante que doc é JSON-safe
    doc = _sanitize_for_json(doc)

    # sempre armazenamos experiment_id dentro do evento (ajuda no pós-processamento)
    payload = {**doc, "ts": time.time(), "experiment_id": experiment_id}

    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    data = np.frombuffer(raw, dtype=np.uint8)

    # lock só para threads desse processo
    with _process_lock:
        with h5py.File(path, "a", libver="latest") as h5:
            h5.swmr_mode = True
            ds = _ensure_events_dataset(h5, key)
            n = ds.shape[0]
            ds.resize((n + 1,))
            ds[n] = data
            h5.flush()


# ============================================================================
# Outputs (blobs e arrays) + metadados
# ============================================================================
def save_output(
    *,
    base_dir: str,
    experiment_id: str,
    kind: str,                 # "image" | "file" | "array" | "text" | "json"
    content: Any,              # bytes | np.ndarray | dict | str
    name: Optional[str] = None,
    mime: Optional[str] = None,
    compression: str = "gzip",
    compression_opts: int = 3,
) -> str:
    """
    Salva um output e retorna um "ref" para linkar no evento.

    kind:
      - "image" / "file" : content deve ser bytes/bytearray
      - "array"          : content deve ser np.ndarray (ou convertível)
      - "text"           : content deve ser str
      - "json"           : content deve ser dict/list/etc (serializável)

    Onde salva:
      - blobs  -> /outputs/blobs/<ref>   (bytes)
      - arrays -> /outputs/arrays/<ref>  (ndarray)
      - meta   -> /outputs/meta/<...>    (JSON em bytes)

    ref:
      - se `name` foi passado: usa `name` (você controla)
      - senão: usa sha256 do conteúdo (dedup automático simples)
    """
    path = shard_path(base_dir, experiment_id)

    # -------------------------
    # Normaliza por tipo
    # -------------------------
    if kind in ("image", "file"):
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError("kind=image/file espera content como bytes/bytearray")
        data_bytes = bytes(content)
        ref = name or _sha256(data_bytes)
        store = "blobs"

    elif kind == "text":
        data_bytes = str(content).encode("utf-8")
        ref = name or _sha256(data_bytes)
        store = "blobs"

    elif kind == "json":
        data_bytes = json.dumps(content, ensure_ascii=False).encode("utf-8")
        ref = name or _sha256(data_bytes)
        store = "blobs"

    elif kind == "array":
        arr = np.asarray(content)
        raw = arr.tobytes(order="C")
        ref = name or _sha256(raw + str(arr.shape).encode() + str(arr.dtype).encode())
        store = "arrays"

    else:
        raise ValueError(f"kind inválido: {kind}")

    # -------------------------
    # Escreve no HDF5
    # -------------------------
    with _process_lock:
        with h5py.File(path, "a", libver="latest") as h5:
            h5.swmr_mode = True
            outputs = h5.require_group("/outputs")
            meta = h5.require_group("/outputs/meta")

            if store == "blobs":
                blobs = outputs.require_group("blobs")

                # evita duplicar se ref já existe
                if ref not in blobs:
                    blobs.create_dataset(
                        ref,
                        data=np.frombuffer(data_bytes, dtype=np.uint8),
                        chunks=True,
                        compression=compression,
                        compression_opts=compression_opts,
                    )

                    meta_obj = {
                        "ref": ref,
                        "kind": kind,
                        "mime": mime,
                        "size": len(data_bytes),
                        "created_ts": time.time(),
                        "name": name,
                        "experiment_id": experiment_id,
                    }

                    meta.create_dataset(
                        ref,
                        data=np.frombuffer(
                            json.dumps(meta_obj, ensure_ascii=False).encode("utf-8"),
                            dtype=np.uint8,
                        ),
                    )

            else:  # arrays
                arrays = outputs.require_group("arrays")

                if ref not in arrays:
                    arrays.create_dataset(
                        ref,
                        data=arr,
                        chunks=True,
                        compression=compression,
                        compression_opts=compression_opts,
                    )

                    meta_obj = {
                        "ref": ref,
                        "kind": "array",
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "created_ts": time.time(),
                        "name": name,
                        "experiment_id": experiment_id,
                    }

                    meta.create_dataset(
                        ref + "__arraymeta",
                        data=np.frombuffer(
                            json.dumps(meta_obj, ensure_ascii=False).encode("utf-8"),
                            dtype=np.uint8,
                        ),
                    )

            h5.flush()

    return ref


# ============================================================================
# Leitura de eventos
# ============================================================================
 
def _read_events_from_h5file(
    h5_path: str,
    collection: str,
    *,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Lê eventos de um único arquivo HDF5.

    Esta função é responsável apenas por abrir um arquivo específico
    (.h5), localizar o dataset de eventos e converter cada registro
    JSON armazenado em um dicionário Python.

    Parâmetros:
    h5_path : str
        Caminho completo do arquivo HDF5 que será lido.

    collection : str
        Nome da coleção (grupo) dentro do HDF5.
        Os eventos são esperados em:
            /<collection>/events

    where : dict, opcional
        Filtro simples no formato:
            {"campo": valor}
        Apenas eventos que possuem exatamente esse valor no campo
        serão retornados.

    Retorno:
    List[Dict[str, Any]]
        Lista de eventos já convertidos para dicionários Python.

    Observações:
    - Cada evento recebe campos auxiliares:
        _index  → posição dentro do dataset
        _file   → arquivo de origem (útil para depuração)
    """

    key = f"/{collection}/events"
    out: List[Dict[str, Any]] = []

    with h5py.File(h5_path, "r", swmr=True) as h5:
        # Se o dataset não existir, retorna lista vazia
        if key not in h5:
            return out

        ds = h5[key]
        n = int(ds.shape[0])

        for i in range(n):
            # Cada evento está armazenado como bytes contendo JSON
            raw = np.array(ds[i], dtype=np.uint8).tobytes().decode(
                "utf-8", errors="replace"
            )

            try:
                ev = json.loads(raw)
            except Exception:
                # Se o JSON estiver corrompido, preserva o conteúdo bruto
                ev = {"_raw": raw}

            # Metadados úteis para depuração
            ev["_index"] = i
            ev["_file"] = h5_path

            # Aplica filtro simples se solicitado
            if where:
                if any(ev.get(k) != v for k, v in where.items()):
                    continue

            out.append(ev)

    return out


def read_events(
    base_dir: str,
    experiment_id: str,
    collection: str,
    *,
    last_n: Optional[int] = None,
    where: Optional[Dict[str, Any]] = None,
    from_all_shards: bool = False,
    prefer_merged: bool = False,
) -> List[Dict[str, Any]]:
    """
    Lê eventos de um experimento, podendo agregar múltiplos shards
    e ordenar os resultados por timestamp.

    Esta função permite três estratégias de leitura:

    1) Arquivo merged (mais rápido para leitura)
    2) Todos os shards do experimento
    3) Apenas o shard do processo atual (modo legado)

    A ordenação final é feita pelo campo:
        ev["ts"]

    Parâmetros 
    base_dir : str
        Diretório onde os arquivos HDF5 do experimento estão armazenados.

    experiment_id : str
        Identificador único do experimento.
        Usado para localizar arquivos como:
            shard_<experiment_id>_pidXXXX.h5
            merged_<experiment_id>.h5

    collection : str
        Nome da coleção dentro do HDF5.

    last_n : int, opcional
        Se informado, retorna apenas os últimos N eventos
        após a ordenação por timestamp.

        Importante:
        O corte é global (após juntar todos os arquivos).

    where : dict, opcional
        Filtro simples:
            {"campo": valor}

    from_all_shards : bool, padrão=False
        Se True:
            Lê todos os shards do experimento.

        Se False:
            Lê apenas o shard do processo atual.

    prefer_merged : bool, padrão=False
        Se True e existir arquivo merged:
            Usa apenas o merged.

        Caso contrário:
            Usa shards.

    Retorno 
    List[Dict[str, Any]]
        Lista de eventos ordenados por timestamp crescente.

    Fluxo interno 
    1) Determina quais arquivos devem ser lidos
    2) Lê eventos de cada arquivo
    3) Junta tudo em memória
    4) Ordena por timestamp
    5) Aplica last_n se necessário

    Observações 
    - Eventos sem campo "ts" são enviados para o final.
    - A ordenação é estável.
    """

    merged_path = default_merged_path(base_dir, experiment_id)
    files_to_read: List[str] = []

    # Escolha da estratégia de leitura
    if prefer_merged and os.path.exists(merged_path):
        # Caminho mais eficiente quando merge já foi realizado
        files_to_read = [merged_path]

    elif from_all_shards:
        # Lê todos os shards existentes
        files_to_read = list_shards(base_dir, experiment_id)

    else:
        # Comportamento antigo: apenas shard local
        files_to_read = [shard_path(base_dir, experiment_id)]

    # Leitura agregada
    out: List[Dict[str, Any]] = []
    for fp in files_to_read:
        if os.path.exists(fp):
            out.extend(
                _read_events_from_h5file(
                    fp,
                    collection,
                    where=where
                )
            )

    # Ordenação global por timestamp
    # Eventos sem "ts" vão para o final
    out.sort(key=lambda ev: float(ev.get("ts", float("inf"))))

    # Corte final
    if last_n is not None:
        last_n = max(0, int(last_n))
        out = out[-last_n:] if last_n > 0 else []

    return out




# ============================================================================
# Merge (juntar shards em um único .h5)
# ============================================================================
def _iter_event_rows(shard_file: str, collection: str) -> Iterable[bytes]:
    """
    Itera sobre todas as linhas (bytes JSON) do dataset /<collection>/events em um shard.
    """
    key = f"/{collection}/events"
    with h5py.File(shard_file, "r") as h5:
        if key not in h5:
            return
        ds = h5[key]
        for row in ds:
            # row é "vlen uint8" -> transforma em bytes
            yield bytes(np.array(row, dtype=np.uint8).tobytes())


def _append_event_row(out_h5: h5py.File, collection: str, raw_json_bytes: bytes) -> None:
    """
    Faz append de uma linha (bytes JSON) no arquivo merged.
    """
    key = f"/{collection}/events"
    ds = _ensure_events_dataset(out_h5, key)
    data = np.frombuffer(raw_json_bytes, dtype=np.uint8)
    n = ds.shape[0]
    ds.resize((n + 1,))
    ds[n] = data


def _copy_dataset_if_missing(dst: h5py.Group, src: h5py.Group, name: str) -> None:
    """
    Copia dataset 'name' de src -> dst se ainda não existir em dst.
    Isso deduplica outputs iguais (mesmo ref).
    """
    if name in dst:
        return
    src.copy(name, dst, name=name)


def _discover_collections_in_shard(shard_file: str) -> List[str]:
    """
    Descobre automaticamente quais coleções existem em um shard.

    Regra:
    - Considera "coleção" todo grupo na raiz que tenha /<collection>/events
    - Ignora /outputs
    """
    cols: set[str] = set()
    with h5py.File(shard_file, "r") as h5:
        for k in h5.keys():  # grupos na raiz
            if k == "outputs":
                continue
            events_key = f"/{k}/events"
            if events_key in h5:
                cols.add(k)
    return sorted(cols)


def _discover_all_collections(shard_files: List[str]) -> List[str]:
    """Une as coleções descobertas em todos os shards."""
    all_cols: set[str] = set()
    for sf in shard_files:
        for c in _discover_collections_in_shard(sf):
            all_cols.add(c)
    return sorted(all_cols)



def merge_experiment_h5(
    *,
    base_dir: str,
    experiment_id: str,
    collections: Optional[List[str]] = None,
    merged_path: Optional[str] = None,
    copy_outputs: bool = True,
    delete_shards_on_success: bool = True,
) -> List[str]:
    """
    Junta todos os shards daquele experiment_id em um único HDF5 final.

    NOVO:
    - Se `collections` for None, ele descobre automaticamente todas as coleções
      existentes nos shards (tudo que tiver /<collection>/events).
    - Se `delete_shards_on_success` for True:
        após gerar e validar o merged, remove os shards do disco.

    Validação (mínima, mas útil):
    - consegue abrir o merged em modo leitura
    - para cada coleção:
        merged tem /<col>/events
        e a contagem de eventos no merged == soma das contagens nos shards
    - se copy_outputs=True:
        merged tem /outputs/meta e (se existiam nos shards) /outputs/blobs e /outputs/arrays
        e a quantidade de chaves (refs) no merged >= união das chaves dos shards

    Retorna:
    - lista de shards usados no merge
    """
    merged_path = merged_path or default_merged_path(base_dir, experiment_id)
    shard_files = list_shards(base_dir, experiment_id)

    if not shard_files:
        raise FileNotFoundError(
            f"Nenhum shard encontrado em {base_dir} para experiment_id={experiment_id}. "
            f"Esperado padrão: shard_{experiment_id}_pid*.h5"
        )

    # Se não foi passado, descobre tudo automaticamente
    if collections is None:
        collections = _discover_all_collections(shard_files)

    os.makedirs(os.path.dirname(merged_path) or ".", exist_ok=True)
 
    # 1) MERGE 
    with h5py.File(merged_path, "w") as out:
        # 1) Merge dos eventos (todas as coleções)
        for col in collections:
            for sf in shard_files:
                for raw in _iter_event_rows(sf, col):
                    _append_event_row(out, col, raw)

        # 2) Merge de outputs (opcional)
        if copy_outputs:
            out_outputs = out.require_group("/outputs")
            out_blobs = out_outputs.require_group("blobs")
            out_arrays = out_outputs.require_group("arrays")
            out_meta = out.require_group("/outputs/meta")

            for sf in shard_files:
                with h5py.File(sf, "r") as src:
                    if "/outputs" not in src:
                        continue

                    if "/outputs/blobs" in src:
                        src_blobs = src["/outputs/blobs"]
                        for ref in src_blobs.keys():
                            _copy_dataset_if_missing(out_blobs, src_blobs, ref)

                    if "/outputs/arrays" in src:
                        src_arrays = src["/outputs/arrays"]
                        for ref in src_arrays.keys():
                            _copy_dataset_if_missing(out_arrays, src_arrays, ref)

                    if "/outputs/meta" in src:
                        src_meta = src["/outputs/meta"]
                        for name in src_meta.keys():
                            _copy_dataset_if_missing(out_meta, src_meta, name)

        out.flush()
 
    # 2) VALIDAÇÃO (só para decidir se pode apagar shards) 
    _validate_merged_h5(
        merged_path=merged_path,
        shard_files=shard_files,
        collections=collections,
        copy_outputs=copy_outputs,
    )
 
    # 3) APAGAR SHARDS (se tudo estiver ok) 
    if delete_shards_on_success:
        for sf in shard_files:
            try:
				# Garante que não está aberto e que o arquivo existe
                if os.path.exists(sf):
                    os.remove(sf)
                    print(f"[MERGE] shard removido: {sf}") 
                else:
                    print(f"[MERGE] shard não encontrado: {sf}")
            except Exception as e:
                raise RuntimeError(f"[MERGE]Falha ao remover shard {sf}: {e}")


    return shard_files


def _validate_merged_h5(
    *,
    merged_path: str,
    shard_files: List[str],
    collections: List[str],
    copy_outputs: bool,
) -> None:
    """
    Valida se o merged parece completo e não corrompido.

    Se falhar, levanta Exception.
    (Nesse caso, merge_experiment_h5 NÃO apaga os shards.)
    """
    # 1) abre o merged (se estiver corrompido, geralmente falha aqui)
    with h5py.File(merged_path, "r") as merged:
        # 2) valida eventos por contagem: merged == soma(shards)
        for col in collections:
            key = f"/{col}/events"

            if key not in merged:
                raise RuntimeError(f"[VALIDATION] merged não contém {key}")

            merged_count = int(merged[key].shape[0])

            shard_sum = 0
            for sf in shard_files:
                with h5py.File(sf, "r") as sh:
                    if key in sh:
                        shard_sum += int(sh[key].shape[0])

            if merged_count != shard_sum:
                raise RuntimeError(
                    f"[VALIDATION] contagem inconsistente em {key}: "
                    f"merged={merged_count} vs soma_shards={shard_sum}"
                )

        # 3) valida outputs (aproximação segura): merged tem pelo menos a união de refs
        if copy_outputs:
            # meta sempre deve existir se copy_outputs=True (mesmo que vazio)
            if "/outputs/meta" not in merged:
                raise RuntimeError("[VALIDATION] merged não contém /outputs/meta")

            # União das chaves em shards
            blobs_union: Set[str] = set()
            arrays_union: Set[str] = set()
            meta_union: Set[str] = set()

            for sf in shard_files:
                with h5py.File(sf, "r") as sh:
                    if "/outputs/blobs" in sh:
                        blobs_union.update(list(sh["/outputs/blobs"].keys()))
                    if "/outputs/arrays" in sh:
                        arrays_union.update(list(sh["/outputs/arrays"].keys()))
                    if "/outputs/meta" in sh:
                        meta_union.update(list(sh["/outputs/meta"].keys()))

            # No merged, se o grupo existir, contamos as chaves
            merged_blobs = set(merged["/outputs/blobs"].keys()) if "/outputs/blobs" in merged else set()
            merged_arrays = set(merged["/outputs/arrays"].keys()) if "/outputs/arrays" in merged else set()
            merged_meta = set(merged["/outputs/meta"].keys()) if "/outputs/meta" in merged else set()

            # O merged precisa conter pelo menos tudo que existia nos shards
            missing_blobs = blobs_union - merged_blobs
            missing_arrays = arrays_union - merged_arrays
            missing_meta = meta_union - merged_meta

            if missing_blobs:
                raise RuntimeError(f"[VALIDATION] faltando blobs no merged: {sorted(list(missing_blobs))[:10]} ...")
            if missing_arrays:
                raise RuntimeError(f"[VALIDATION] faltando arrays no merged: {sorted(list(missing_arrays))[:10]} ...")
            if missing_meta:
                raise RuntimeError(f"[VALIDATION] faltando meta no merged: {sorted(list(missing_meta))[:10]} ...")
