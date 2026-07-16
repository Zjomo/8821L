program Mono_DLL_Test;

uses
  Forms,
  Mono_DLL_TestU in 'Mono_DLL_TestU.pas' {TestForm},
  ARC_SpectraProU in '..\Delphi_Advanced_Functions\ARC_SpectraProU.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TTestForm, TestForm);
  Application.Run;
end.
