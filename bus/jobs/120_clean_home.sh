set -uo pipefail
H="$HOME"
D=/data2/chenyuxiang/_from_home
mkdir -p "$D"

echo "########## what is in \$HOME right now ##########"
du -sh "$H"/* "$H"/.[!.]* 2>/dev/null | sort -h | tail -25

echo
echo "########## moving things WE created out of \$HOME ##########"
# ~/bus/ came from Mac-side commands pasted into the server shell by mistake.
# The runner never looked at it; it is dead weight.
if [ -d "$H/bus" ]; then
  mv "$H/bus" "$D/bus_stray_$(date +%s)" && echo "moved ~/bus -> $D/"
fi

# Any tarballs we uploaded to home.
for f in "$H"/myoicl_*.tar.gz "$H"/*.tar.gz; do
  [ -e "$f" ] || continue
  mv "$f" "$D/" && echo "moved $(basename "$f") -> $D/"
done

# The HTTPS token file. SSH-over-443 is what actually authenticates now, so
# this is both unused and a secret sitting in plain text.
if [ -e "$H/.git-credentials" ]; then
  rm -f "$H/.git-credentials" && echo "removed ~/.git-credentials (unused; SSH key is what authenticates)"
fi
git -C /data2/chenyuxiang/code/myoicl config --unset credential.helper 2>/dev/null && echo "unset credential.helper"

# Scratch we made under /tmp during the v4.1 rollout.
rm -rf /tmp/v41 2>/dev/null && echo "removed /tmp/v41"

echo
echo "########## \$HOME after ##########"
du -sh "$H" 2>/dev/null
du -sh "$H"/* "$H"/.[!.]* 2>/dev/null | sort -h | tail -15
echo
echo "NOTE: ~/.ssh/ is left alone on purpose -- ssh only reads keys from there,"
echo "      and the whole channel to GitHub depends on it. It is a few KB."
echo
echo "########## confirming our real footprint is all under /data2 ##########"
du -sh /data2/chenyuxiang/code/myoicl /data2/chenyuxiang/runs 2>/dev/null
