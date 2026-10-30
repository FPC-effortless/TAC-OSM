"""Archive-boundary falsifiers; no fits or held-out scoring."""
import hashlib,io,zipfile
import pytest
from scripts.replay_langmem086087 import safe,archive_bytes,sampler

def test_archive_checksum_before_member_read(tmp_path):
 p=tmp_path/'payload.zip';p.write_bytes(b'not an archive')
 with pytest.raises(ValueError,match='checksum'):archive_bytes(tmp_path,{'path':p.name,'sha256':'0'*64,'bytes':14})

def test_paths_and_symlinks_rejected(tmp_path):
 for p in ('../outside','/absolute','a/../b','a\\b'):
  with pytest.raises(ValueError):safe(tmp_path,p)
 (tmp_path/'link').symlink_to(tmp_path/'missing')
 with pytest.raises(ValueError):safe(tmp_path,'link')

def test_duplicate_archive_members_rejected(tmp_path):
 p=tmp_path/'duplicate.zip'
 with zipfile.ZipFile(p,'w') as z:
  z.writestr('MODEL.pt',b'one')
  with pytest.warns(UserWarning):z.writestr('MODEL.pt',b'two')
 raw=p.read_bytes()
 with pytest.raises(ValueError,match='duplicate'):archive_bytes(tmp_path,{'path':p.name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
