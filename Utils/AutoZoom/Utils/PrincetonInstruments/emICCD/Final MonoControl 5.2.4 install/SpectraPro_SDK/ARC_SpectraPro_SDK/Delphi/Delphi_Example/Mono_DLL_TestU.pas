unit Mono_DLL_TestU;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, StdCtrls, Spin, ARC_SpectraProU, ExtCtrls;

type
  TTestForm = class(TForm)
    FindPortBtn: TButton;
    DialogList: TListBox;
    EnumNumSpin: TSpinEdit;
    OpenBtn: TButton;
    CloseBtn: TButton;
    ConnectedChk: TCheckBox;
    SendCMDEdt: TEdit;
    SendCMDLabel: TLabel;
    SendCMDBtn: TButton;
    MonoInfoBtn: TButton;
    Label1: TLabel;
    SetWaveEdt: TEdit;
    WaveUnitRadio: TRadioGroup;
    CenterWaveEdt: TEdit;
    CenterWaveLabel: TLabel;
    SetWaveBtn: TButton;
    TurretSpin: TSpinEdit;
    GratingSpin: TSpinEdit;
    SetTurretBtn: TButton;
    SetGratingBtn: TButton;
    SetSlitWidthBtn: TButton;
    HomeSlitBtn: TButton;
    NewSlitWidthEdt: TEdit;
    NewFilterEdt: TEdit;
    SetFilterBtn: TButton;
    HomeFilterBtn: TButton;
    SlitSpin: TSpinEdit;
    NewWaveLabel: TLabel;
    procedure FormCreate(Sender: TObject);
    procedure FindPortBtnClick(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure EnumNumSpinChange(Sender: TObject);
    procedure OpenBtnClick(Sender: TObject);
    procedure CloseBtnClick(Sender: TObject);
    procedure SendCMDBtnClick(Sender: TObject);
    procedure MonoInfoBtnClick(Sender: TObject);
    procedure SetWaveBtnClick(Sender: TObject);
    procedure SetTurretBtnClick(Sender: TObject);
    procedure SetGratingBtnClick(Sender: TObject);
    procedure SetSlitWidthBtnClick(Sender: TObject);
    procedure HomeSlitBtnClick(Sender: TObject);
    procedure SetFilterBtnClick(Sender: TObject);
    procedure HomeFilterBtnClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  TestForm: TTestForm;

implementation

{$R *.dfm}

procedure TTestForm.FormCreate(Sender: TObject);
begin
top  := 0;
left := 0;

if Dynamic_Load_ARC_SpectraPro_DLL
   then DialogList.Items.Add('Loaded ARC_Spectra_Pro.dll')
   else DialogList.Items.Add('Error : Failed to load ARC_Spectra_Pro.dll');
end;

procedure TTestForm.FindPortBtnClick(Sender: TObject);
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

procedure TTestForm.FormDestroy(Sender: TObject);
begin
unload_ARC_SpectraPro_DLL;
end;

procedure TTestForm.EnumNumSpinChange(Sender: TObject);
begin
try
// indicate if the current enum is open or not
ConnectedChk.checked := ARC_Valid_Mono_Enum(EnumNumSpin.Value);
except
   ConnectedChk.checked := false;
   end;
end;

procedure TTestForm.OpenBtnClick(Sender: TObject);
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

procedure TTestForm.CloseBtnClick(Sender: TObject);
begin
// close the mono port
if not ARC_Close_Mono(EnumNumSpin.Value)
       then showmessage('Error : Failed to Close Port');
// update if the port is or is not opened
EnumNumSpinChange(Sender);
end;

procedure TTestForm.SendCMDBtnClick(Sender: TObject);
var
ReturnStr : string;
begin
screen.cursor := crHourGlass;
try
// send a command to the instrument, note the function adds the required CR (^M)
// to the function, so the user does not need to. Placing a CR in the command
// string can have unpredictable results.
if not ARC_Send_CMD_To_Mono(EnumNumSpin.Value,SendCMDEdt.text,ReturnStr,5000)
   then showmessage('Error : Failed to Send Command');
SendCMDLabel.Caption := 'Returned : ' +ReturnStr;
except
   end;
screen.cursor := crDefault;
end;

procedure TTestForm.MonoInfoBtnClick(Sender: TObject);
var
outstr : string;
outdbl : double;
outint : integer;
outbool: wordbool;
loop   : integer;
wkstr  : string;
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

if ARC_get_Mono_Focallength(EnumNumSpin.Value,outdbl)
then begin
     DialogList.Items.add('FocalLength : ' +format('%.2f',[outdbl]));
     end
else DialogList.Items.add('Error : No Focal Length');

if ARC_get_Mono_HalfAngle(EnumNumSpin.Value,outdbl)
then begin
     DialogList.Items.add('Half Angle : ' +format('%.2f',[(180.0 * outdbl) / 3.14]));
     end
else DialogList.Items.add('Error : No HalfAngle');

if ARC_get_Mono_DetectorAngle(EnumNumSpin.Value,outdbl)
then begin
     DialogList.Items.add('Det Angle : ' +format('%.2f',[(180.0 * outdbl) / 3.14]));
     end
else DialogList.Items.add('Error : No Detector Angle');

if ARC_get_Mono_Turret_Gratings(EnumNumSpin.Value,outint)
then begin
     DialogList.Items.add('Gratings per Turret : ' +IntToStr(outint));
     end
else DialogList.Items.add('Error : Gratings per Turret');

if ARC_get_Mono_Double(EnumNumSpin.Value,outbool)
then begin
     if outbool
        then DialogList.Items.add('Double Mono : True')
        else DialogList.Items.add('Double Mono : False');
     end
else DialogList.Items.add('Error : Double');

if ARC_get_Mono_Wavelength_nm(EnumNumSpin.Value,outdbl)
then begin
     DialogList.Items.add('Wavelength : ' +Format('%.4f',[outdbl]) +' nm');
     end
else DialogList.Items.add('Error : Wavelength nm');
if ARC_get_Mono_Wavelength_ang(EnumNumSpin.Value,outdbl)
then begin
     DialogList.Items.add('Wavelength : ' +Format('%.3f',[outdbl]) +' A');
     end
else DialogList.Items.add('Error : Wavelength Angstrom');
if ARC_get_Mono_Wavelength_eV(EnumNumSpin.Value,outdbl)
then begin
     DialogList.Items.add('Wavelength : ' +Format('%.4f',[outdbl]) +' eV');
     end
else DialogList.Items.add('Error : Wavelength eV');
if ARC_get_Mono_Wavelength_micron(EnumNumSpin.Value,outdbl)
then begin
     DialogList.Items.add('Wavelength : ' +Format('%.4f',[outdbl]) +' microns');
     end
else DialogList.Items.add('Error : Wavelength micron');
if ARC_get_Mono_Wavelength_absCM(EnumNumSpin.Value,outdbl)
then begin
     DialogList.Items.add('Wavelength : ' +Format('%.1f',[outdbl]) +' Absolute CM-1');
     end
else DialogList.Items.add('Error : Wavelength Absolute WaveNumber');
if ARC_get_Mono_Wavelength_relCM(EnumNumSpin.Value,546.0, outdbl)
then begin
     DialogList.Items.add('Wavelength : ' +Format('%.1f',[outdbl]) +' Relative CM-1');
     end
else DialogList.Items.add('Error : Wavelength Relative WaveNumber');

if ARC_get_Mono_Turret(EnumNumSpin.Value,outint)
then begin
     DialogList.Items.add('Turret : ' +IntToStr(outint));
     end
else DialogList.Items.add('Error : Turret');
if ARC_get_Mono_Grating(EnumNumSpin.Value,outint)
then begin
     DialogList.Items.add('Grating : ' +IntToStr(outint));
     end
else DialogList.Items.add('Error : Grating');
if ARC_get_Mono_Wavelength_Cutoff_nm(EnumNumSpin.Value,outdbl)
then begin
     DialogList.Items.add('Max Wave : ' +format('%.0f',[outdbl]) +' nm');
     end
else DialogList.Items.add('Error : Max Wave');
if ARC_get_Mono_Wavelength_Min_nm(EnumNumSpin.Value,outdbl)
then begin
    DialogList.Items.add('Min Wave : ' +format('%.0f',[outdbl]) +' nm');
     end
else DialogList.Items.add('Error : Min Wave');
for loop := 1 to 15
 do begin
    if ARC_get_Mono_Grating_Installed(EnumNumSpin.Value,loop)
    then begin
         wkStr := 'Grating ' +IntToStr(loop) +' : ';
         if ARC_get_Mono_Grating_Density(EnumNumSpin.Value,loop,outint)
            and
            ARC_get_Mono_Grating_Blaze(EnumNumSpin.Value,loop,outstr)
         then begin
              DialogList.Items.add(wkStr +IntToStr(outint) +' g/mm, Blaze = ' +outstr);
              end
         else DialogList.Items.add('Error : ' +wkStr);
         end;
    end;
// get diverters and slits
for loop := 1 to 2
 do begin
    if ARC_get_Mono_diverter_Pos_Str(EnumNumSpin.Value,loop,outstr)
    then begin
         DialogList.Items.add('Mirror ' +IntToStr(Loop) +' on : ' +outstr);
         end
    else DialogList.Items.add('Error : Mirror not Motorized');
    ARC_Mono_Slit_Name_Str((loop * 2) - 1, wkstr);
    if ARC_get_Mono_Slit_Type_Str(EnumNumSpin.Value,(loop * 2) - 1,outstr)
       then DialogList.Items.add(wkStr +' Slit : ' +outStr)
       else DialogList.Items.add('Error : ' +wkStr);
    if ARC_get_Mono_Slit_Width(EnumNumSpin.Value,(loop * 2) - 1,outint)
       then DialogList.Items.add('Slit Witdh : ' +IntToStr(outint));
    ARC_Mono_Slit_Name_Str((loop * 2), wkstr);
    if ARC_get_Mono_Slit_Type_Str(EnumNumSpin.Value,(loop * 2),outstr)
       then DialogList.Items.add(wkStr +' Slit : ' +outStr)
       else DialogList.Items.add('Error : ' +wkStr);
    if ARC_get_Mono_Slit_Width(EnumNumSpin.Value,(loop * 2),outint)
       then DialogList.Items.add('Slit Witdh : ' +IntToStr(outint));
    end;
ARC_get_Mono_Double(EnumNumSpin.Value,outbool);
if outbool // slave present
then begin
     for loop := 3 to 4
      do begin
         if ARC_get_Mono_diverter_Pos_Str(EnumNumSpin.Value,loop,outstr)
         then begin
              DialogList.Items.add('Mirror ' +IntToStr(Loop) +' on : ' +outstr);
              end
         else DialogList.Items.add('Error : Mirror not Motorized');
         ARC_Mono_Slit_Name_Str((loop * 2) - 1, wkstr);
         if ARC_get_Mono_Slit_Type_Str(EnumNumSpin.Value,(loop * 2) - 1,outstr)
            then DialogList.Items.add(wkStr +' Slit : ' +outStr)
            else DialogList.Items.add('Error : ' +wkStr);
         if ARC_get_Mono_Slit_Width(EnumNumSpin.Value,(loop * 2) - 1,outint)
            then DialogList.Items.add('Slit Witdh : ' +IntToStr(outint));
         ARC_Mono_Slit_Name_Str((loop * 2), wkstr);
         if ARC_get_Mono_Slit_Type_Str(EnumNumSpin.Value,(loop * 2),outstr)
            then DialogList.Items.add(wkStr +' Slit : ' +outStr)
            else DialogList.Items.add('Error : ' +wkStr);
         if ARC_get_Mono_Slit_Width(EnumNumSpin.Value,(loop * 2),outint)
            then DialogList.Items.add('Slit Witdh : ' +IntToStr(outint));
         end;
     end;
// get filter info
if ARC_get_Mono_Filter_Present(EnumNumSpin.Value)
then begin
     if ARC_get_Mono_Filter_Position(EnumNumSpin.Value,outint)
        then DialogList.Items.add('Filter Position : ' +IntToStr(outInt))
        else DialogList.Items.add('Error : Filter Position');
     end
else begin
     DialogList.Items.add('No Filter');
     end;
end;

procedure TTestForm.SetWaveBtnClick(Sender: TObject);
var
WaveDbl : double;
CentDbl : double;
begin
screen.cursor := crHourGlass;
try
// convert the text to a new wavelength
wavedbl := StrToFloat(trim(SetWaveEdt.Text));
// based on the radio for wavelength units set the wavelength
case WaveUnitRadio.ItemIndex of
     0 : if not ARC_set_Mono_Wavelength_ang(EnumNumSpin.Value,wavedbl) then showmessage('Error : Request rejected');
     1 : if not ARC_set_Mono_Wavelength_nm(EnumNumSpin.Value,wavedbl) then showmessage('Error : Request rejected');
     2 : if not ARC_set_Mono_Wavelength_micron(EnumNumSpin.Value,wavedbl) then showmessage('Error : Request rejected');
     3 : if not ARC_set_Mono_Wavelength_eV(EnumNumSpin.Value,wavedbl) then showmessage('Error : Request rejected');
     4 : if not ARC_set_Mono_Wavelength_absCM(EnumNumSpin.Value,wavedbl) then showmessage('Error : Request rejected');
     5 : try
         CentDbl := StrToFloat(trim(CenterWaveEdt.Text));
         if not ARC_set_Mono_Wavelength_relCM(EnumNumSpin.Value,CentDbl,wavedbl) then showmessage('Error : Request rejected');
         except
            showmessage('Error : Malformed center wavelength');
            end;
     else showmessage('Error : no wave units selected');
     end;
except
   showmessage('Error : Malformed wavelength');
   end;
screen.cursor := crDefault;
end;

procedure TTestForm.SetTurretBtnClick(Sender: TObject);
begin
screen.cursor := crHourGlass;
try
if not ARC_set_Mono_Turret(EnumNumSpin.Value,TurretSpin.Value)
   then showmessage('Error : Failed to set turret');
except
   end;
screen.cursor := crDefault;
end;

procedure TTestForm.SetGratingBtnClick(Sender: TObject);
begin
screen.cursor := crHourGlass;
try
if not ARC_set_Mono_Grating(EnumNumSpin.Value,GratingSpin.Value)
   then showmessage('Error : Failed to set Grating');
except
   end;
screen.cursor := crDefault;
end;

procedure TTestForm.SetSlitWidthBtnClick(Sender: TObject);
begin
screen.cursor := crHourGlass;
try
if not ARC_set_Mono_Slit_Width(EnumNumSpin.Value,SlitSpin.Value,round(StrToFloat(Trim(NewSlitWidthEdt.Text))))
   then showmessage('Error : Failed to set Slit Width');
except
   end;
screen.cursor := crDefault;
end;

procedure TTestForm.HomeSlitBtnClick(Sender: TObject);
begin
screen.cursor := crHourGlass;
try
if not ARC_Mono_Slit_Home(EnumNumSpin.Value,SlitSpin.Value)
   then showmessage('Error : Failed to Home Slit');
except
   end;
screen.cursor := crDefault;
end;

procedure TTestForm.SetFilterBtnClick(Sender: TObject);
begin
screen.cursor := crHourGlass;
try
if not ARC_set_Mono_Filter_Position(EnumNumSpin.Value,round(StrToFloat(Trim(NewFilterEdt.Text))))
   then showmessage('Error : Failed to set Filter Position');
except
   end;
screen.cursor := crDefault;
end;

procedure TTestForm.HomeFilterBtnClick(Sender: TObject);
begin
screen.cursor := crHourGlass;
try
if not ARC_Mono_Filter_Home(EnumNumSpin.Value)
   then showmessage('Error : Failed to Home Filter');
except
   end;
screen.cursor := crDefault;
end;

end.
