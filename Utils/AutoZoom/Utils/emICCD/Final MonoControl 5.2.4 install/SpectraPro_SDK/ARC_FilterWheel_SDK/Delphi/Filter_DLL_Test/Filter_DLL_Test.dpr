program Filter_DLL_Test;

uses
  Forms,
  Filter_DLL_TestU in 'Filter_DLL_TestU.pas' {FilterTestForm},
  ARC_FilterWheelU in 'ARC_FilterWheelU.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TFilterTestForm, FilterTestForm);
  Application.Run;
end.
