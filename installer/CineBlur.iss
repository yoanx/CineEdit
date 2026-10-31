; CineBlur 安装包脚本（Inno Setup 6）
; 编译：ISCC.exe installer\CineBlur.iss
;
; 设计取舍
;   · PrivilegesRequired=lowest —— 装到 %LOCALAPPDATA%，全程不弹 UAC，
;     朋友拿到就能装，不需要管理员口令
;   · Compression=lzma2/max 但不启用 SolidCompression —— 载荷 4.6 GB，
;     固态压缩需要数 GB 内存，编译机内存不足时会中途失败
;   · 不额外安装运行库：torch 的 CUDA DLL、VC++ 运行时、ffmpeg、
;     RAFT 权重全部已在 dist 内，装完即可离线使用

#define AppName "CineBlur"
#define AppVersion "1.0.0"
#define AppPublisher "CineBlur"
#define AppExeName "CineBlur.exe"

[Setup]
AppId={{9C4B7E2A-5D31-4A8E-B6F2-73A1C0D95E48}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\CineBlur
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=out
OutputBaseFilename=CineBlur-Setup-{#AppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} 运动模糊渲染器
Compression=lzma2/max
SolidCompression=no
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
Source: "..\dist\CineBlur\*"; DestDir: "{app}"; \
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
