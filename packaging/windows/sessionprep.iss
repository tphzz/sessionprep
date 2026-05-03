; SessionPrep Windows Installer (unsigned Inno Setup entrypoint)
;
; Build from repo root:
;   ISCC /DAPP_VERSION=x.y.z /DDIST_DIR=dist_nuitka /DARCH_SUFFIX=win-x64 packaging\windows\sessionprep.iss

#define SESSIONPREP_SIGNED_BUILD 0
#include "sessionprep-common.isinc"
