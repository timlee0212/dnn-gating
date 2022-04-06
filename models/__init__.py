from .pvt import *

import os
import requests
import re

def getFilename_fromCd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]

pretrain_weight_list = {
    "pvt_tiny" : "https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth",
    "pvt_small": "https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth",
    "pvt_medium": "https://github.com/whai362/PVT/releases/download/v2/pvt_medium.pth",
    "pvt_large": "https://github.com/whai362/PVT/releases/download/v2/pvt_large.pth"
}
ckpt_folder = os.path.join(os.path.abspath(__file__),".." ,"_ckpt")
if not os.path.exists(ckpt_folder):
    os.mkdir(ckpt_folder)
for name, url in pretrain_weight_list.items():
    if not os.path.exists(os.path.join(ckpt_folder, name + ".pth")):
        r = requests.get(url, allow_redirects=True)
        filename = getFilename_fromCd(r.headers.get('content-disposition'))
        open(os.path.join(ckpt_folder, filename), 'wb').write(r.content)
        #os.system("wget {0}".format(url))