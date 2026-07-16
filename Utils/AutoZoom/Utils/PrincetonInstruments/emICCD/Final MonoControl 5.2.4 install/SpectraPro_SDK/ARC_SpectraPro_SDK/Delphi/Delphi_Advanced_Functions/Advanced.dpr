program Advanced;

uses
  Forms,
  AdvancedU in 'AdvancedU.pas' {Advanced_FunctionsF};

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TAdvanced_FunctionsF, Advanced_FunctionsF);
  Application.Run;
end.
