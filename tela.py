import PySimpleGUI as sg

class TelaPython:
    def __init__(self):
        # Layout
        layout = [
            [sg.Text('URL'), sg.Input()],
            [sg.Button('Enviar')]
        ]
        # Janela
        janela = sg.Window("Insira a URL:").layout(layout)
        # Extrair os dados da tela
        self.button, self.values = janela.Read()

    def Iniciar(self):
        print(self.values)


tela = TelaPython()
tela.Iniciar()
