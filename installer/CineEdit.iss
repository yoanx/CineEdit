; CineEdit 安装包脚本（Inno Setup 6）
; 编译：ISCC.exe installer\CineEdit.iss
;
; 设计取舍
;   · PrivilegesRequired=lowest —— 装到 %LOCALAPPDATA%，全程不弹 UAC，
;     朋友拿到就能装，不需要管理员口令
;   · Compression=lzma2/ultra64 + SolidCompression=yes —— 载荷 4.8 GB，
;     固实压缩把安装包压到 2 GB 以内（GitHub Release 单文件上限）
;   · 不额外安装运行库：torch 的 CUDA DLL、VC++ 运行时、ffmpeg、
;     RAFT 权重全部已在 dist 内，装完即可离线使用

#define AppName "CineEdit"
#define AppVersion "1.0.0"
#define AppPublisher "CineEdit"
#define AppExeName "CineEdit.exe"

[Setup]
AppId={{9C4B7E2A-5D31-4A8E-B6F2-73A1C0D95E48}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\CineEdit
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=out
OutputBaseFilename=CineEdit-Setup-{#AppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} 运动模糊渲染器
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
CloseApplications=yes
RestartApplications=no
AllowNoIcons=yes
VersionInfoVersion=1.0.0.0
VersionInfoProductName={#AppName}
VersionInfoDescription=运动模糊渲染器 安装程序
VersionInfoCompany={#AppPublisher}

[Languages]
Name: "zh"; MessagesFile: "ChineseSimplified.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\CineEdit\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 程序自身写入的临时文件（如果有）留在这里清理；用户视频与缓存不动
Type: filesandordirs; Name: "{app}\_internal\__pycache__"
