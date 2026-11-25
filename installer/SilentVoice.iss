; SilentVoice.iss — Інсталятор та деінсталятор (Inno Setup 6) для SilentVoice TTS
#define AppName "SilentVoice TTS"
#define AppVersion "1.0.3"
#define AppPublisher "SilentVoice"
#define AppExeName "SilentVoice.exe"
#define AppMutexName "SilentVoiceTTS_Mutex"

[Setup]
AppId={{9F2B7C4E-8A1D-4E53-B694-1C7D2F5A8E03}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={commonpf}\SilentVoice
DefaultGroupName={#AppName}
AllowNoIcons=no
DirExistsWarning=no
DisableProgramGroupPage=no
OutputDir=..\dist\installer
OutputBaseFilename=SilentVoice_Setup_{#AppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
#ifdef FastBuild
Compression=lzma2/fast
SolidCompression=no
#else
Compression=lzma2/ultra64
SolidCompression=yes
#endif
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=6
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
UsedUserAreasWarning=no
WizardStyle=modern
DisableWelcomePage=no
ShowLanguageDialog=auto
AppMutex={#AppMutexName}
CloseApplications=yes
RestartApplications=no
#if Pos(".", AppVersion) == 0
VersionInfoVersion={#AppVersion}.0
#else
VersionInfoVersion={#AppVersion}
#endif
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Installer
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}

[Languages]
Name: "uk"; MessagesFile: "compiler:Languages\Ukrainian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "pl"; MessagesFile: "compiler:Languages\Polish.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[CustomMessages]
uk.DesktopIconDesc=Створити ярлик на робочому столі
uk.AutostartDesc=Запускати {#AppName} автоматично разом із Windows (згорнутим у трей)
uk.LaunchApp=Запустити {#AppName}
uk.RemoveUserDataPrompt=Видалити також усі налаштування користувача, швидкі фрази, сесію Telegram та завантажені локальні TTS-моделі (%APPDATA%\SilentVoice)?

en.DesktopIconDesc=Create a desktop shortcut
en.AutostartDesc=Start {#AppName} automatically with Windows (minimized to tray)
en.LaunchApp=Launch {#AppName}
en.RemoveUserDataPrompt=Do you also want to remove all user settings, quick phrases, Telegram session, and downloaded local TTS models (%APPDATA%\SilentVoice)?

pl.DesktopIconDesc=Utwórz skrót na pulpicie
pl.AutostartDesc=Uruchamiaj {#AppName} automatycznie wraz z systemem Windows (zminimalizowany w zasobniku)
pl.LaunchApp=Uruchom {#AppName}
pl.RemoveUserDataPrompt=Czy usunąć także wszystkie ustawienia użytkownika, szybkie frazy, sesję Telegram oraz pobrane lokalne modele TTS (%APPDATA%\SilentVoice)?

de.DesktopIconDesc=Desktop-Verknüpfung erstellen
de.AutostartDesc={#AppName} automatisch mit Windows starten (im Infobereich minimiert)
de.LaunchApp={#AppName} starten
de.RemoveUserDataPrompt=Möchten Sie auch alle Benutzereinstellungen, Schnellphrasen, die Telegram-Sitzung und heruntergeladene lokale TTS-Modelle (%APPDATA%\SilentVoice) entfernen?

fr.DesktopIconDesc=Créer un raccourci sur le Bureau
fr.AutostartDesc=Lancer {#AppName} automatiquement au démarrage de Windows (réduit dans la barre d'état)
fr.LaunchApp=Lancer {#AppName}
fr.RemoveUserDataPrompt=Voulez-vous également supprimer tous les paramètres utilisateur, phrases rapides, session Telegram et modèles TTS locaux téléchargés (%APPDATA%\SilentVoice) ?

es.DesktopIconDesc=Crear un acceso directo en el escritorio
es.AutostartDesc=Iniciar {#AppName} automáticamente con Windows (minimizado en la bandeja)
es.LaunchApp=Iniciar {#AppName}
es.RemoveUserDataPrompt=¿Desea eliminar también toda la configuración del usuario, frases rápidas, sesión de Telegram y los modelos TTS locales descargados (%APPDATA%\SilentVoice)?

ru.DesktopIconDesc=Создать ярлык на рабочем столе
ru.AutostartDesc=Запускать {#AppName} автоматически вместе с Windows (свёрнутым в трей)
ru.LaunchApp=Запустить {#AppName}
ru.RemoveUserDataPrompt=Удалить также все пользовательские настройки, быстрые фразы, сессию Telegram и скачанные локальные TTS-модели (%APPDATA%\SilentVoice)?

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIconDesc}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "autostart"; Description: "{cm:AutostartDesc}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\SilentVoice\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; PyInstaller stores the onedir bundle's libraries and data in _internal.
; Replace that directory as a unit so obsolete DLLs/modules cannot remain after
; an update. User settings and downloaded models live under %APPDATA%\SilentVoice.
[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"; IconFilename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "SilentVoiceTTS"; ValueData: """{app}\{#AppExeName}"" --minimized"; Flags: uninsdeletevalue; Tasks: autostart

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchApp}"; Flags: nowait postinstall skipifsilent
Filename: "{app}\{#AppExeName}"; Flags: nowait skipifnotsilent; Check: ShouldAutoRelaunch

[Code]
function OpenEventW(dwDesiredAccess: LongWord; bInheritHandle: Boolean; lpName: String): THandle;
  external 'OpenEventW@kernel32.dll stdcall';
function SetEvent(hEvent: THandle): Boolean;
  external 'SetEvent@kernel32.dll stdcall';
function CloseHandle(hObject: THandle): Boolean;
  external 'CloseHandle@kernel32.dll stdcall';

function ShouldAutoRelaunch(): Boolean;
begin
  Result := (ExpandConstant('{param:AUTORELAUNCH|0}') = '1');
end;

procedure InitializeWizard();
var
  CurDirLower: String;
  LocalAppLower: String;
begin
  CurDirLower := LowerCase(WizardForm.DirEdit.Text);
  LocalAppLower := LowerCase(ExpandConstant('{localappdata}'));
  // Якщо в реєстрі залишився старий шлях у AppData\Local\Programs, перемикаємо на C:\Program Files\SilentVoice
  if (LocalAppLower <> '') and (Pos(LocalAppLower, CurDirLower) > 0) then
  begin
    WizardForm.DirEdit.Text := ExpandConstant('{commonpf}\SilentVoice');
  end;
end;

procedure SignalGracefulQuit();
var
  EvHandle: THandle;
  WaitIdx: Integer;
begin
  if not CheckForMutexes('{#AppMutexName}') then
    Exit;
  EvHandle := OpenEventW($0002, False, 'SilentVoiceTTS_QuitEvent');
  if EvHandle <> 0 then
  begin
    SetEvent(EvHandle);
    CloseHandle(EvHandle);
    for WaitIdx := 1 to 12 do
    begin
      if not CheckForMutexes('{#AppMutexName}') then
        Break;
      Sleep(150);
    end;
  end;
end;

procedure CloseRunningApp();
var
  ResultCode: Integer;
  WaitIdx: Integer;
  WasRunning: Boolean;
begin
  WasRunning := CheckForMutexes('{#AppMutexName}');
  if WasRunning then
    SignalGracefulQuit();

  if CheckForMutexes('{#AppMutexName}') then
  begin
    Exec('taskkill.exe', '/F /T /IM {#AppExeName}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    Exec('taskkill.exe', '/F /T /IM llama-server.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    Exec('taskkill.exe', '/F /T /IM llama-tts.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    for WaitIdx := 1 to 10 do
    begin
      if not CheckForMutexes('{#AppMutexName}') then
        Break;
      Sleep(250);
    end;
    Sleep(400);
  end
  else if WasRunning then
  begin
    // Коротка пауза, щоб Windows повністю звільнила файли після коректного виходу програми
    Sleep(300);
  end;
end;

function InitializeSetup(): Boolean;
begin
  // Автоматично закриваємо запущену програму ще до перевірки AppMutex інсталятором
  CloseRunningApp();
  Result := True;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  CloseRunningApp();
  Result := '';
end;

function InitializeUninstall(): Boolean;
begin
  CloseRunningApp();
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir: String;
  UserDataDir: String;
  PurgeParam: String;
  ShouldPurge: Boolean;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    // Завжди прибираємо запис автозапуску (навіть якщо його увімкнули всередині програми)
    RegDeleteValue(HKEY_CURRENT_USER, 'Software\Microsoft\Windows\CurrentVersion\Run', 'SilentVoiceTTS');

    // Додатково очищуємо папку програми {app}, якщо окремі DLL були ще заблоковані в момент основного кроку
    AppDir := ExpandConstant('{app}');
    if DirExists(AppDir) then
    begin
      Sleep(400);
      DelTree(AppDir, True, True, True);
    end;

    UserDataDir := ExpandConstant('{userappdata}\SilentVoice');
    if DirExists(UserDataDir) then
    begin
      PurgeParam := ExpandConstant('{param:PURGE|0}');
      ShouldPurge := (PurgeParam = '1');
      if (not ShouldPurge) and (not UninstallSilent) then
      begin
        ShouldPurge := (MsgBox(CustomMessage('RemoveUserDataPrompt'), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES);
      end;
      if ShouldPurge then
      begin
        DelTree(UserDataDir, True, True, True);
      end;
    end;
  end;
end;
