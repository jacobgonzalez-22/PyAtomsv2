; PyAtoms Windows Installer 
; Created on Wed Sep 02 17:38:41 PM
; @author: Jacob
;
; Modification Log
; ------------------
; 2026-09-03 - Jacob Gonzalez
;   - Updated installer version to 1.0.3
;
; 2026-09-02 - Jacob Gonzalez
;   - Added Inno Setup configuration for standalne Windows installation

#define MyAppName "PyAtoms"
#define MyAppVersion "1.0.3"
#define MyAppExeName "PyAtoms.exe"

[Setup]
AppId={{A7E54B0E-5F6E-4A3B-B2B8-63F4D1807B29}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=GutierrezPhys
AppPublisherURL=https://github.com/GutierrezPhys/PyAtoms
AppSupportURL=https://github.com/GutierrezPhys/PyAtoms/issues
AppUpdatesURL=https://github.com/GutierrezPhys/PyAtoms
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest

SourceDir=..\..\standalone_dist\PyAtoms
OutputDir=..\..\installer_output
OutputBaseFilename=PyAtoms-Setup-{#MyAppVersion}

Compression=lzma2
SolidCompression=yes 
WizardStyle=modern 

UninstallDisplayName={#MyAppName} 
UninstallDisplayIcon={app}\{#MyAppExeName} 

[Tasks] 
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked 

[Files] 
Source: "*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs 

[Icons] 
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}" 
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon 

[Run] 
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent