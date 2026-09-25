from pathlib import Path
import hashlib,json,tarfile,datetime
b=Path('/Users/yuanzhiyi/Desktop/MetaDrive_PIVOT_20260923/followup_precision/complete_backup_20260924')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
registered_source_manifest_sha256=sha(b.parent/'SOURCE_PROTOCOL_SHA256SUMS.txt')
registered_source_manifest_matches=registered_source_manifest_sha256=='92455d8d2d5536fbeba69c238a42beb473ea244dec03c7246f96ea3a67a5e4cb'
metadata=json.loads((b/'server_archive_metadata.json').read_text())
archive=b/'metadrive_precision_20260923_complete.tar.gz';manifest=b/'METADRIVE_PRECISION_SERVER_SHA256SUMS.txt'
actual_archive_sha=sha(archive);actual_manifest_sha=sha(manifest)
expected={}
for line in manifest.read_text().splitlines():
 h,n=line.split('  ',1);expected[n]=h
seen=set();mismatches=[];unexpected=[];duplicates=[];filecount=bytecount=0;eps={'calibration':0,'confirmation':0}
with tarfile.open(archive,'r|gz') as tf:
 for member in tf:
  if not member.isfile():unexpected.append({'name':member.name,'type':'not_regular_file'});continue
  name=member.name
  if name in seen:duplicates.append(name)
  seen.add(name);filecount+=1;bytecount+=member.size
  stream=tf.extractfile(member);h=hashlib.sha256()
  for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
  if name not in expected:unexpected.append(name)
  elif h.hexdigest()!=expected[name]:mismatches.append(name)
  parts=Path(name).parts
  if len(parts)>2 and parts[0] in eps and parts[1]=='episodes':eps[parts[0]]+=1
missing=sorted(set(expected)-seen)
report={'checked_local':datetime.datetime.now().astimezone().isoformat(),'archive':str(archive),'registered_source_manifest_sha256':registered_source_manifest_sha256,'registered_source_manifest_matches':registered_source_manifest_matches,'archive_bytes':archive.stat().st_size,'archive_sha256':actual_archive_sha,'archive_sha256_matches_server':actual_archive_sha==metadata['archive_sha256'],'manifest_sha256':actual_manifest_sha,'manifest_sha256_matches_server':actual_manifest_sha==metadata['manifest_sha256'],'verified_file_count':filecount,'verified_uncompressed_file_bytes':bytecount,'expected_file_count':len(expected),'episode_file_counts':eps,'mismatched_files':mismatches,'missing_files':missing,'unexpected_members':unexpected,'duplicate_members':duplicates,'verification_method':'Stream each compressed archive file; recompute each SHA256 against server manifest without expanding all episodes on Desktop.','pass':registered_source_manifest_matches and actual_archive_sha==metadata['archive_sha256'] and actual_manifest_sha==metadata['manifest_sha256'] and not(mismatches or missing or unexpected or duplicates) and filecount==metadata['file_count'] and bytecount==metadata['uncompressed_file_bytes'] and eps==metadata['episode_file_counts']}
(b/'backup_verification.json').write_text(json.dumps(report,indent=2)+'\n')
if not report['pass']:raise RuntimeError(json.dumps(report))
(b/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(b.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'))
print(json.dumps(report,indent=2))
