unit AdvancedU;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, StdCtrls, Spin, ARC_SpectraProU, ExtCtrls;

type
  TAdvanced_FunctionsF = class(TForm)
    ConnectedChk: TCheckBox;
    FindPortBtn: TButton;
    EnumNumSpin: TSpinEdit;
    OpenBtn: TButton;
    CloseBtn: TButton;
    MonoInfoBtn: TButton;
    DialogList: TListBox;
    MonoIntLedChk: TCheckBox;
    MoveStepsEdt: TEdit;
    StepLabel: TLabel;
    MoveStepsBtn: TButton;
    RestoreFactoryBtn: TButton;
    ResetMonoBtn: TButton;
    JogBtn: TButton;
    DirRadio: TRadioGroup;
    JogWaveNMLabel: TLabel;
    JogRateRadio: TRadioGroup;
    ScanRateEdt: TEdit;
    ScanRateLabel: TLabel;
    SetScanRateBtn: TButton;
    StartScanEdt: TEdit;
    StopScanEdt: TEdit;
    StartScanLabel: TLabel;
    StopScanLAbel: TLabel;
    ScanBtn: TButton;
    ScanWaveNMLabel: TLabel;
    MonoInitGratEdt: TEdit;
    MonoInitWaveEdt: TEdit;
    MonoInitScanEdt: TEdit;
    MonoInitGratBtn: TButton;
    MonoInitWaveBtn: TButton;
    MonoInitScanBtn: TButton;
    GratLabel: TLabel;
    GroovesLabel: TLabel;
    BlazeStr: TLabel;
    UnInstallGratButton: TButton;
    InstallGratButton: TButton;
    GratSpin: TSpinEdit;
    GroovesEdit: TEdit;
    BlazeEdit: TEdit;
    BlazeGroup: TRadioGroup;
    DisplayWaveEdit: TEdit;
    RefWaveEdit: TEdit;
    CalcOffSetButton: TButton;
    CalcGAdjButton: TButton;
    NewOffsetLabel: TLabel;
    NewGAdjLabel: TLabel;
    SetOffsetButton: TButton;
    SetGAdjButton: TButton;
    DisplayWaveLabel: TLabel;
    RefWaveLabel: TLabel;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FindPortBtnClick(Sender: TObject);
    procedure EnumNumSpinChange(Sender: TObject);
    procedure OpenBtnClick(Sender: TObject);
    procedure CloseBtnClick(Sender: TObject);
    procedure MonoInfoBtnClick(Sender: TObject);
    procedure MonoIntLedChkClick(Sender: TObject);
    procedure MoveStepsBtnClick(Sender: TObject);
    procedure RestoreFactoryBtnClick(Sender: TObject);
    procedure ResetMonoBtnClick(Sender: TObject);
    procedure JogBtnMouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure JogBtnMouseUp(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure SetScanRateBtnClick(Sender: TObject);
    procedure ScanBtnClick(Sender: TObject);
    procedure MonoInitGratBtnClick(Sender: TObject);
    procedure MonoInitWaveBtnClick(Sender: TObject);
    procedure MonoInitScanBtnClick(Sender: TObject);
    procedure InstallGratButtonClick(Sender: TObject);
    procedure UnInstallGratButtonClick(Sender: TObject);
    procedure CalcOffSetButtonClick(Sender: TObject);
    procedure SetOffsetButtonClick(Sender: TObject);
    procedure CalcGAdjButtonClick(Sender: TObject);
    procedure SetGAdjButtonClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  StopJog : boolean;
  Advanced_FunctionsF: TAdvanced_FunctionsF;

implementation

{$R *.dfm}

procedure TAdvanced_FunctionsF.FormCreate(Sender: TObject);
begin
top  := 0;
left := 0;

if Dynamic_Load_ARC_SpectraPro_DLL
   then DialogList.Items.Add('Loaded ARC_Spectra_Pro.dll')
   else DialogList.Items.Add('Error : Failed to load ARC_Spectra_Pro.dll');
end;

procedure TAdvanced_FunctionsF.FormDestroy(Sender: TObject);
begin
unload_ARC_SpectraPro_DLL;
end;

procedure TAdvanced_FunctionsF.FindPortBtnClick(Sender: TObject);
var
Num_Mono : integer;
model_loop : integer;
model_str  : string;
begin
screen.cursor := crHourGlass;
try
DialogList.Items.clear;

if ARC_Search_For_Mono(Num_Mono)
then begin
     for model_loop := 0 to Num_Mono - 1
      do begin
         if ARC_get_Mono_preOpen_Model(model_loop,model_str)
            then DialogList.Items.Add('Enum : ' +IntToStr(model_loop) +' ' +model_str);
         end;
     end
else begin
     DialogList.Items.Add('No Monos attached');
     end;
finally
   screen.cursor := crDefault;
   end;
end;

procedure TAdvanced_FunctionsF.EnumNumSpinChange(Sender: TObject);
begin
try
// indicate if the current enum is open or not
ConnectedChk.checked := ARC_Valid_Mono_Enum(EnumNumSpin.Value);
except
   ConnectedChk.checked := false;
   end;
end;

procedure TAdvanced_FunctionsF.OpenBtnClick(Sender: TObject);
var
dummy : integer;
begin
screen.cursor := crHourGlass;
try
// open the mono port
if not ARC_Open_Mono(EnumNumSpin.Value,dummy)
       then showmessage('Error : Failed to Open Port');
// update if the port is or is not opened
EnumNumSpinChange(Sender);
finally
   screen.cursor := crDefault;
   end;
end;

procedure TAdvanced_FunctionsF.CloseBtnClick(Sender: TObject);
begin
// close the mono port
if not ARC_Close_Mono(EnumNumSpin.Value)
       then showmessage('Error : Failed to Close Port');
// update if the port is or is not opened
EnumNumSpinChange(Sender);
end;

procedure TAdvanced_FunctionsF.MonoInfoBtnClick(Sender: TObject);
var
outstr : string;
outdbl : double;
outint : integer;
outint2: integer;
//outbool: wordbool;
loop   : integer;
//wkstr  : string;
begin
DialogList.Items.Clear;
if ARC_get_Mono_Model(EnumNumSpin.Value,outstr)
then begin
     DialogList.Items.add('Model : ' +outstr);
     end
else DialogList.Items.add('Error : No Model');
if ARC_get_Mono_Serial(EnumNumSpin.Value,outstr)
then begin
     DialogList.Items.add('Serial : ' +outstr);
     end
else DialogList.Items.add('Error : No Serial');
// grating info
outint2 := 0;
for loop := 1 to 15
 do begin
    if ARC_get_Mono_Grating_Offset(EnumNumSpin.Value,loop,outint)
       and
       ARC_get_Mono_Grating_GAdjust(EnumNumSpin.Value,loop,outint2)
    then  DialogList.Items.add('Grating' +IntToStr(loop)
                               +' Offset : ' +IntToStr(outint)
                               +' Gadjust : ' +IntToStr(outint2));
    end;
// mono startup settings
if ARC_get_Mono_Init_Grating(EnumNumSpin.Value,outint)
   then DialogList.Items.add('Init Grating : ' +IntToStr(outint))
   else DialogList.Items.add('Error : Init Grating');
if ARC_get_Mono_Init_Wave_nm(EnumNumSpin.Value,outdbl)
   then DialogList.Items.add('Init Wavelength : ' +Format('%.2f',[outdbl]) +' nm')
   else DialogList.Items.add('Error : Init Wavelength');
if ARC_get_Mono_Init_ScanRate_nm(EnumNumSpin.Value,outdbl)
   then DialogList.Items.add('Init Scan Rate : ' +Format('%.2f',[outdbl]) +' nm')
   else DialogList.Items.add('Error : Init ScanRate');
if ARC_get_Mono_Scan_Rate_nm_min(EnumNumSpin.Value,outdbl)
   then DialogList.Items.add('Scan Rate : ' +Format('%.2f',[outdbl]) +' nm/min')
   else DialogList.Items.add('Error : Scan Rate nm/min');
// mono interrupter settings
if ARC_get_Mono_Int_Led_On(EnumNumSpin.Value)
then begin
     DialogList.Items.add('Int Led : On');
     if ARC_get_Mono_Motor_Int(EnumNumSpin.Value)
        then DialogList.Items.add('Motor Int : Open')
        else DialogList.Items.add('Motor Int : Closed');
     if ARC_get_Mono_Wheel_Int(EnumNumSpin.Value)
        then DialogList.Items.add('Wheel Int : Open')
        else DialogList.Items.add('Wheel Int : Closed');
     end
else DialogList.Items.add('Int Led : Off');
end;

procedure TAdvanced_FunctionsF.MonoIntLedChkClick(Sender: TObject);
begin
Screen.Cursor := crHourGlass;
try
if MonoIntLedChk.Checked
then begin
     if not ARC_set_Mono_Int_Led(EnumNumSpin.Value,true) then showmessage('Error : Failed to Turn on Int LED');
     end
else begin
     if not ARC_set_Mono_Int_Led(EnumNumSpin.Value,false) then showmessage('Error : Failed to Turn off Int LED');
     end;
finally
   Screen.Cursor := crDefault;
   end;
end;

procedure TAdvanced_FunctionsF.MoveStepsBtnClick(Sender: TObject);
var
Num_Steps : integer;
Old_nm    : double;
New_nm    : double;
begin
Screen.Cursor := crHourGlass;
try
Num_Steps := round(StrToFloat(Trim(MoveStepsEdt.Text)));
// obtain the current wavelength in nm before we move, so we can display it later
ARC_get_Mono_Wavelength_nm(EnumNumSpin.Value,Old_nm);
// now move the number of steps requested
if not ARC_Mono_Move_Steps(EnumNumSpin.Value,Num_Steps)
then showmessage('Error : Request rejected')
else begin
     // display what has just happened ...
     DialogList.Items.Clear;
     DialogList.Items.Add('PreMove Wavelength : ' +Format('%.3f',[Old_nm]) +' nm');
     DialogList.Items.Add('Steps Moved : ' +IntToStr(Num_Steps));
     // obtain post move nm
     ARC_get_Mono_Wavelength_nm(EnumNumSpin.Value,New_nm);
     DialogList.Items.Add('PostMove Wavelength : ' +Format('%.3f',[New_nm]) +' nm');
     end;
except
   showmessage('Error : Malformed Number of Steps');
   end;
Screen.Cursor := crDefault;
end;

procedure TAdvanced_FunctionsF.RestoreFactoryBtnClick(Sender: TObject);
begin
screen.Cursor := crHourGlass;
try
if not ARC_Mono_Restore_Factory_Settings(EnumNumSpin.Value)
   then showmessage('Error : Request rejected');
finally
   screen.cursor := crDefault;
   end;
end;

procedure TAdvanced_FunctionsF.ResetMonoBtnClick(Sender: TObject);
begin
screen.Cursor := crHourGlass;
try
if not ARC_Mono_Reset(EnumNumSpin.Value)
   then showmessage('Error : Request rejected');
finally
   screen.cursor := crDefault;
   end;
end;

procedure TAdvanced_FunctionsF.JogBtnMouseDown(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
var
Scan_Done : boolean;
Cur_Wave  : double;
MaxRate   : boolean;
ScanUp    : boolean;
begin
StopJog := false;
if DirRadio.ItemIndex = 0
   then ScanUp := true
   else ScanUp := false;
if JogRateRadio.ItemIndex = 0
   then MaxRate := true
   else MaxRate := false;

if ARC_Mono_Start_Jog (EnumNumSpin.Value,MaxRate,ScanUp)
then begin
     Scan_Done := false;
     Cur_Wave := 0.0;
     while (not StopJog)
            and
           (ARC_Mono_Scan_Done(EnumNumSpin.Value,Scan_Done,Cur_Wave))
        do begin
           JogWaveNMLabel.Caption := format('%.3f',[Cur_Wave]) +' nm';
           JogWaveNMLabel.Repaint;
           application.processmessages; // give the application events a chance
                                        // to catch up
           end;
     end
else showmessage('Error : Unable to Scan');
ARC_Mono_Stop_Jog (EnumNumSpin.Value);
// now display the final wavelength
if ARC_get_Mono_Wavelength_nm(EnumNumSpin.Value,Cur_Wave)
   then JogWaveNMLabel.Caption := format('%.3f',[Cur_Wave]) +' nm';
end;

procedure TAdvanced_FunctionsF.JogBtnMouseUp(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
StopJog := true;
end;

procedure TAdvanced_FunctionsF.SetScanRateBtnClick(Sender: TObject);
var
Scan_Rate : double;
begin
Screen.Cursor := crHourGlass;
try
Scan_Rate := StrToFloat(Trim(ScanRateEdt.text));
if not ARC_set_Mono_Scan_Rate_nm_min (EnumNumSpin.Value,Scan_Rate)
   then ShowMessage('Error : Setting Scan Rate');
except
   showmessage('Error : Malformed Scan Rate');
   end;
Screen.Cursor := crDefault;
end;

procedure TAdvanced_FunctionsF.ScanBtnClick(Sender: TObject);
var
Start_Wave_nm : double;
Stop_Wave_nm  : double;
Scan_Done     : boolean;
Cur_Wave      : double;
begin
try
Start_Wave_nm := StrToFloat(Trim(StartScanEdt.text));
Stop_Wave_nm  := StrToFloat(Trim(StopScanEdt.text));
if ARC_set_Mono_Wavelength_nm(EnumNumSpin.Value,Start_Wave_nm) // set initial wavelength
then if ARC_Mono_Start_Scan_To_nm(EnumNumSpin.Value,Stop_Wave_nm) // start scanning
     then begin
          Scan_Done := false;
          Cur_Wave := 0.0;
          while (not Scan_Done)
                 and
                // see if we are done, this function also updates the motor
                // speed so that it matches scan rate. It needs to be called
                // regularly until the scan is done.
                (ARC_Mono_Scan_Done(EnumNumSpin.Value,Scan_Done,Cur_Wave))
             do begin
                ScanWaveNMLabel.Caption := format('%.3f',[Cur_Wave]) +' nm';
                ScanWaveNMLabel.Repaint;
                application.processmessages;
                end;
          // needed to make sure the wavelength is correctly set
          ARC_Mono_Stop_Jog (EnumNumSpin.Value);
          // when done scanning read the final wavelength
          if ARC_get_Mono_Wavelength_nm(EnumNumSpin.Value,Cur_Wave)
             then ScanWaveNMLabel.Caption := format('%.3f',[Cur_Wave]) +' nm';
          end
     else showmessage('Error : Unable to Scan')
else showmessage('Error : Unable to set Start Wavelength');
except
   showmessage('Error : MalFormed Data');
   end;
end;

procedure TAdvanced_FunctionsF.MonoInitGratBtnClick(Sender: TObject);
var
Init_Grat : integer;
begin
Screen.Cursor := crHourGlass;
try
Init_Grat := round(StrToFloat(Trim(MonoInitGratEdt.Text)));
if not ARC_set_Mono_Init_Grating(EnumNumSpin.Value,Init_Grat)
   then showmessage('Error : Request rejected');
except
   showmessage('Error : Malformed Init Grating');
   end;
Screen.Cursor := crDefault;
end;

procedure TAdvanced_FunctionsF.MonoInitWaveBtnClick(Sender: TObject);
var
Init_Wave : double;
begin
Screen.Cursor := crHourGlass;
try
Init_Wave := StrToFloat(Trim(MonoInitWaveEdt.Text));
if not ARC_get_Mono_Init_Wave_nm(EnumNumSpin.Value,Init_Wave)
   then showmessage('Error : Request rejected');
except
   showmessage('Error : Malformed Init Wavelength');
   end;
Screen.Cursor := crDefault;
end;

procedure TAdvanced_FunctionsF.MonoInitScanBtnClick(Sender: TObject);
var
Init_Rate : double;
begin
Screen.Cursor := crHourGlass;
try
Init_Rate := StrToFloat(Trim(MonoInitScanEdt.Text));
if not ARC_set_Mono_Init_ScanRate_nm(EnumNumSpin.Value,Init_Rate)
   then showmessage('Error : Request rejected');
except
   showmessage('Error : Malformed Init Scan Rate');
   end;
Screen.Cursor := crDefault;
end;

procedure TAdvanced_FunctionsF.InstallGratButtonClick(Sender: TObject);
var
Density  : Integer;
Blaze    : String;
NMBlaze  : boolean;
HOLBlaze : boolean;
MIRROR   : boolean;
begin
try
Density := Round(StrToFloat(Trim(GroovesEdit.text)));
Blaze   := trim(BlazeEdit.Text);
case BlazeGroup.ItemIndex of
     0 : begin
         MIRROR   := true;
         NMBlaze  := false;
         HOLBlaze := false;
         end;
     1 : begin
         MIRROR   := false;
         NMBlaze  := true;
         HOLBlaze := false;
         end;
     2 : begin
         MIRROR   := false;
         NMBlaze  := false;
         HOLBlaze := false;
         end;
     3 : begin 
         MIRROR   := false;
         NMBlaze  := false;
         HOLBlaze := true;
         end;
     else begin
          MIRROR   := false;
          NMBlaze  := false;
          HOLBlaze := false;
          end;
     end;
if not ARC_Mono_Grating_Install(EnumNumSpin.Value,GratSpin.Value,Density,Blaze,NMBlaze,HOLBlaze,MIRROR)
   then showmessage('Error : Unable to install grating');
except
   showmessage('Error : MalFormed Data');
   end;
end;

procedure TAdvanced_FunctionsF.UnInstallGratButtonClick(Sender: TObject);
begin
try
if not ARC_Mono_Grating_UnInstall(EnumNumSpin.Value,GratSpin.Value)
   then showmessage('Error : Failed to Uninstall Grating');
except
   end;
end;

procedure TAdvanced_FunctionsF.CalcOffSetButtonClick(Sender: TObject);
var
newWave    : double;
newRefWave : double;
newOffset  : integer;
begin
try
newWave    := StrToFloat(Trim(DisplayWaveEdit.text));
newRefWave := StrToFloat(Trim(RefWaveEdit.text));
if ARC_Mono_Grating_Calc_Offset(EnumNumSpin.Value,GratSpin.Value,newWave,newRefWave,newOffset)
then begin
     NewOffsetLabel.Caption := 'New Offset = ' +IntToStr(newOffset);
     end;
except
   end;
end;

procedure TAdvanced_FunctionsF.SetOffsetButtonClick(Sender: TObject);
var
newWave    : double;
newRefWave : double;
newOffset  : integer;
begin
try
newWave    := StrToFloat(Trim(DisplayWaveEdit.text));
newRefWave := StrToFloat(Trim(RefWaveEdit.text));
if ARC_Mono_Grating_Calc_Offset(EnumNumSpin.Value,GratSpin.Value,newWave,newRefWave,newOffset)
then begin
     NewOffsetLabel.Caption := 'New Offset = ' +IntToStr(newOffset);
     if not ARC_set_Mono_Grating_Offset(EnumNumSpin.Value,GratSpin.Value,newOffset)
        then showmessage('Error : Failed to set offset!');
     end;
except
   end;
end;

procedure TAdvanced_FunctionsF.CalcGAdjButtonClick(Sender: TObject);
var
newWave    : double;
newRefWave : double;
newGAdjust : integer;
begin
try
newWave    := StrToFloat(Trim(DisplayWaveEdit.text));
newRefWave := StrToFloat(Trim(RefWaveEdit.text));
if ARC_Mono_Grating_Calc_Gadjust(EnumNumSpin.Value,GratSpin.Value,newWave,newRefWave,newGAdjust)
then begin
     NewGAdjLabel.Caption := 'New GAdjust = ' +IntToStr(newGAdjust);
     end;
except
   end;
end;

procedure TAdvanced_FunctionsF.SetGAdjButtonClick(Sender: TObject);
var
newWave    : double;
newRefWave : double;
newGAdjust : integer;
begin
try
newWave    := StrToFloat(Trim(DisplayWaveEdit.text));
newRefWave := StrToFloat(Trim(RefWaveEdit.text));
if ARC_Mono_Grating_Calc_Gadjust(EnumNumSpin.Value,GratSpin.Value,newWave,newRefWave,newGAdjust)
then begin
     NewGAdjLabel.Caption := 'New GAdjust = ' +IntToStr(newGAdjust);
     if not ARC_set_Mono_Grating_Gadjust(EnumNumSpin.Value,GratSpin.Value,newGAdjust)
        then showmessage('Error : Failed to set G Adjust!');
     end;
except
   end;
end;

end.
