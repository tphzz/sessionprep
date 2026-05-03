; SessionPrep Windows Installer (signed Inno Setup entrypoint)
;
; Build from repo root, with signtool.exe available on PATH:
;   ISCC '/SsessionprepSignTool=signtool.exe sign /a /fd SHA256 /t http://time.certum.pl/ $f' /DAPP_VERSION=x.y.z /DDIST_DIR=dist_nuitka /DARCH_SUFFIX=win-x64 packaging\windows\sessionprep-signed.iss

#define SESSIONPREP_SIGNED_BUILD 1
#include "sessionprep-common.isinc"
