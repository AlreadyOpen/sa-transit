"""Adelaide Metro vehicle positions (GTFS-realtime, CC BY 4.0): one GET of /v1/realtime/vehicle_positions per run (15 min).
Bucket (overwritten in place):  live/vehicles.json               the latest feed decoded to JSON, every field (including the
                                                                 feed's own TfNSW vehicle extension: air_conditioned,
                                                                 wheelchair_accessible, under vehicle.vehicle)
                                live/vehicles.pb                 the latest feed as received (protobuf)
                                live/vehicles-YYYY-MM-DD.parquet that UTC day's snapshots so far, appended each run (GeoParquet: one
                                                                 row per vehicle per snapshot, `entity` = the full entity as JSON)
kaggle_archive.py moves each complete month to its own Kaggle dataset, and only then removes its days from the bucket.
On any error the run is skipped (exit 0 with a warning): no retry loop. Needs HF_TOKEN. Usage: python3 live.py"""
import datetime, json, os, sys, tempfile, urllib.request
import duckdb
from google.protobuf import unknown_fields
from google.protobuf.json_format import MessageToDict
from google.transit import gtfs_realtime_pb2
from huggingface_hub import batch_bucket_files, create_bucket, download_bucket_files

URL = 'https://gtfs.adelaidemetro.com.au/v1/realtime/vehicle_positions'
UA = {'User-Agent': 'AlreadyOpen sa-transit (https://github.com/AlreadyOpen/sa-transit)'}
REPO = 'alreadyopen/sa-transit'
EXT = 1999                                       # TfnswVehicleDescriptor on VehicleDescriptor (adelaidemetro_gtfsr.proto)
ROWS = """
SELECT to_timestamp(snapshot) AS "snapshot", e->>'id' AS "entity_id",
  e->'vehicle'->'trip'->>'trip_id' AS "trip_id", e->'vehicle'->'trip'->>'route_id' AS "route_id",
  (e->'vehicle'->'trip'->>'direction_id')::INTEGER AS "direction_id", e->'vehicle'->'trip'->>'start_date' AS "start_date",
  e->'vehicle'->'trip'->>'start_time' AS "start_time", e->'vehicle'->'trip'->>'schedule_relationship' AS "schedule_relationship",
  e->'vehicle'->'vehicle'->>'id' AS "vehicle_id", e->'vehicle'->'vehicle'->>'label' AS "vehicle_label",
  e->'vehicle'->'vehicle'->>'license_plate' AS "license_plate",
  (e->'vehicle'->'vehicle'->'tfnsw_vehicle_descriptor'->>'air_conditioned')::BOOLEAN AS "air_conditioned",
  (e->'vehicle'->'vehicle'->'tfnsw_vehicle_descriptor'->>'wheelchair_accessible')::INTEGER AS "wheelchair_accessible",
  (e->'vehicle'->'position'->>'latitude')::DOUBLE AS "lat", (e->'vehicle'->'position'->>'longitude')::DOUBLE AS "lon",
  (e->'vehicle'->'position'->>'bearing')::DOUBLE AS "bearing", (e->'vehicle'->'position'->>'speed')::DOUBLE AS "speed",
  (e->'vehicle'->'position'->>'odometer')::DOUBLE AS "odometer",
  (e->'vehicle'->>'current_stop_sequence')::INTEGER AS "current_stop_sequence", e->'vehicle'->>'stop_id' AS "stop_id",
  e->'vehicle'->>'current_status' AS "current_status", to_timestamp((e->'vehicle'->>'timestamp')::BIGINT) AS "timestamp",
  e->'vehicle'->>'congestion_level' AS "congestion_level", e->'vehicle'->>'occupancy_status' AS "occupancy_status",
  (e->'vehicle'->>'occupancy_percentage')::INTEGER AS "occupancy_percentage", e AS "entity",
  ST_Point((e->'vehicle'->'position'->>'longitude')::DOUBLE, (e->'vehicle'->'position'->>'latitude')::DOUBLE) AS "geometry"
FROM (SELECT (json->'header'->>'timestamp')::BIGINT AS snapshot, unnest(json_extract(json, '$.entity[*]')) AS e
      FROM read_json_objects('{src}'))"""


def varints(b):
    """a length-delimited sub-message of varint fields -> {field number: value}"""
    out, i = {}, 0
    while i < len(b):
        vals = []
        for _ in range(2):                                             # the tag, then the value
            v = s = 0
            while True:
                x = b[i]; i += 1; v |= (x & 0x7f) << s; s += 7
                if x < 0x80: break
            vals.append(v)
        assert vals[0] & 7 == 0, 'extension holds a non-varint field'
        out[vals[0] >> 3] = vals[1]
    return out


def decode(body):
    """protobuf -> dict with every field, the TfNSW extension merged in; unknown fields anywhere else become a warning"""
    f = gtfs_realtime_pb2.FeedMessage(); f.ParseFromString(body)
    d = MessageToDict(f, preserving_proto_field_name=True)
    def walk(m, path):
        for u in unknown_fields.UnknownFieldSet(m):
            if not (path.endswith('.vehicle.vehicle') and u.field_number == EXT):
                print(f'::warning::unknown field {u.field_number} at {path}: not decoded (only in vehicles.pb)')
        for fd, v in m.ListFields():
            if fd.message_type:
                for i, x in enumerate(v if fd.is_repeated else [v]): walk(x, f'{path}.{fd.name}')
    walk(f, 'feed')
    for e, ed in zip(f.entity, d.get('entity', [])):
        if e.HasField('vehicle') and e.vehicle.HasField('vehicle'):
            for u in unknown_fields.UnknownFieldSet(e.vehicle.vehicle):
                if u.field_number == EXT:
                    x = varints(u.data)
                    ed['vehicle']['vehicle']['tfnsw_vehicle_descriptor'] = {
                        **({'air_conditioned': bool(x[1])} if 1 in x else {}), **({'wheelchair_accessible': x[2]} if 2 in x else {})}
    return d


def main():
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, headers=UA), timeout=60) as r:
            body = r.read()
        d = decode(body)
        assert d.get('header', {}).get('timestamp'), f'no header timestamp: {str(d)[:200]}'
    except Exception as e:                                             # 5xx, timeout, bad protobuf: wait for the next run
        print(f'::warning::Adelaide Metro vehicle_positions: {e}; skipping this run'); sys.exit(0)
    day = datetime.datetime.fromtimestamp(int(d['header']['timestamp']), datetime.UTC).date()
    name = f'live/vehicles-{day}.parquet'
    w = tempfile.mkdtemp()
    snap, pb, old, out = (os.path.join(w, x) for x in ('vehicles.json', 'vehicles.pb', 'old.parquet', 'new.parquet'))
    json.dump(d, open(snap, 'w'), separators=(',', ':')); open(pb, 'wb').write(body)
    create_bucket(REPO, private=False, exist_ok=True)                    # the first snapshot may run before the first bake
    download_bucket_files(REPO, [(name, old)], raise_on_missing_files=False)   # absent on the day's first run
    con = duckdb.connect(); con.execute('INSTALL spatial; LOAD spatial')
    new = ROWS.format(src=snap)
    q = f"SELECT * FROM read_parquet('{old}') UNION ALL BY NAME {new}" if os.path.exists(old) else new
    if d.get('entity'):
        con.execute(f"COPY ({q} ORDER BY snapshot, entity_id) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    rows = con.execute(f"SELECT count(*), count(DISTINCT snapshot) FROM '{out}'").fetchone() if os.path.exists(out) else (0, 0)
    add = [(snap, 'live/vehicles.json'), (pb, 'live/vehicles.pb')] + ([(out, name)] if os.path.exists(out) else [])
    batch_bucket_files(REPO, add=add)
    print(json.dumps({'snapshot': d['header']['timestamp'], 'vehicles': len(d.get('entity', [])), 'file': name, 'rows': rows[0], 'snapshots': rows[1]}))


if __name__ == '__main__':
    if sys.argv[1:] == ['test']:                                       # the extension decoder, no network
        assert varints(bytes([0x08, 0x01, 0x10, 0x01])) == {1: 1, 2: 1} and varints(bytes([0x10, 0xac, 0x02])) == {2: 300}
        print('live: self-test ok')
    elif len(sys.argv) > 1:                                            # decode a saved feed, no network: python3 live.py feed.pb
        print(json.dumps(decode(open(sys.argv[1], 'rb').read()), indent=1)[:3000])
    else:
        main()
