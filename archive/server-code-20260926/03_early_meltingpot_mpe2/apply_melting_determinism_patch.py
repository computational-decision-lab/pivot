"""Versioned, narrowly scoped native-order patch for the adaptive extension.

Preserves original Lua source alongside its manifest. No edits to reward
formulas, agents, adaptation steps, maps, or configured random draws. Ordering of avatar
identity and equal-priority updater dispatch becomes explicit. This is a
runtime extension, not an unchanged official benchmark release.
"""
import argparse, hashlib, json
from pathlib import Path

def sha(x):return hashlib.sha256(x).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--stage',choices=['avatars','updaters'],required=True);p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();module=a.source/'meltingpot/lua/modules';a.evidence.mkdir(parents=True,exist_ok=True)
    name='base_simulation.lua' if a.stage=='avatars' else 'updater_registry.lua'
    path=module/name;before=path.read_bytes();s=before.decode()
    if a.stage=='avatars':
        assert 'local function orderedAvatarPairs' not in s
        needle="local BaseSimulation = class.Class()"
        assert needle in s
        helper='''-- Adaptive-extension reproducibility patch: stable player identity order.
local function orderedAvatarPairs(avatars)
  local keys = {}
  for key, _ in pairs(avatars) do table.insert(keys, key) end
  table.sort(keys, function(a, b)
    return avatars[a]:getComponent('Avatar'):getIndex() <
           avatars[b]:getComponent('Avatar'):getIndex()
  end)
  local i = 0
  return function()
    i = i + 1
    local key = keys[i]
    if key ~= nil then return key, avatars[key] end
  end
end

'''
        s=s.replace(needle,helper+needle)
        assert s.count('pairs(self._variables.avatarObjects)')==5
        s=s.replace('pairs(self._variables.avatarObjects)','orderedAvatarPairs(self._variables.avatarObjects)')
    else:
        needle='''    for name, _ in pairs(updaterNames) do
      table.insert(updateOrder, name)
    end'''
        assert s.count(needle)==1
        replacement='''    -- Adaptive-extension reproducibility patch: stable equal-priority order.
    local names = {}
    for name, _ in pairs(updaterNames) do table.insert(names, name) end
    table.sort(names)
    for _, name in ipairs(names) do table.insert(updateOrder, name) end'''
        s=s.replace(needle,replacement)
    backup=a.evidence/(name+'.original');assert not backup.exists();backup.write_bytes(before)
    path.write_text(s)
    manifest={'patch':a.stage,'relative_path':str(path.relative_to(a.source)),'before_sha256':sha(before),'after_sha256':sha(path.read_bytes()),
              'purpose':'Make unordered native table iteration deterministic; not a change to rewards or learned policies.',
              'scientific_status':'Must pass fresh native replay before new mechanism cohort; historical results retained unchanged.'}
    (a.evidence/(a.stage+'.json')).write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
