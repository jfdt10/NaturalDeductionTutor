import re
from graphviz import Digraph
import os

class DeducaoInvalidaError(Exception):
    def __init__(self, passo, regra, mensagem):
        self.passo = passo
        self.regra = regra
        self.mensagem = mensagem
    
    def __str__(self):
        return f"Erro no passo {self.passo} ({self.regra}): {self.mensagem}"

# Função para normalizar expressões
def normalizar(expr):
    """
    Remove espaços, parênteses redundantes e normaliza o símbolo de disjunção.
    """
    expr = expr.replace("v", "∨")  # Substitui "v" por "∨"
    return expr.replace(" ", "").replace("(", "").replace(")", "")

# Função para comparar proposições normalizadas
def proposicoes_iguais(expr1, expr2):
    return normalizar(expr1) == normalizar(expr2)

# Função para converter LaTeX para formato legível
def converter_latex_para_legivel(entrada):
    entrada = entrada.replace(r"\(","").replace(r"\)", "").strip()
    substituicoes = {
        r"\\rightarrow": "→",
        r"\\land": "∧",
        r"\\lor": "∨",
        r"\\neg": "¬",
        r"\\vdash": "⊢",
        r"\\leftrightarrow": "↔",
    }
    for latex, simbolo in substituicoes.items():
        entrada = re.sub(latex, simbolo, entrada)
    return entrada.strip()

# Função para extrair a conclusão de um passo
def extrair_conclusao(conteudo):
    """
    Extrai a conclusão de um passo, removendo justificativas e regras lógicas do final.
    """
    conteudo = re.sub(r'\((Hipótese|MP|SP|CJ|MT|SD|SH|DN|AD|vE|↔I|↔E|PC|RAA|Hip-PC|Hip-RAA|COM|DMOR|COND)\)', '', conteudo).strip()
    conteudo = re.sub(r'\s*(?:MP|SP|CJ|MT|SD|SH|DN|AD|vE|↔I|↔E|PC|RAA|Hip-PC|Hip-RAA|COM|DMOR|COND)\(\d+(?:,\d+|-\d+)*\)$', '', conteudo).strip()
    if conteudo.startswith('"') and conteudo.endswith('"'):
        conteudo = conteudo[1:-1].replace('"', '')
        conteudo = f'"{conteudo}"'
    else:
        conteudo = conteudo.replace('"', '')
    return conteudo

# Função para coletar as relações entre os passos
def coletar_relacoes(passos_dict, feedback, conclusao):
    """
    Analisa cada passo da dedução, identifica a regra aplicada e constrói uma lista de tuplas
    contendo as informações sobre as relações entre os passos.
    Verifica as dependências de passos anteriores para todas as regras.
    """
    relacoes = []
    validade_passos = {passo: True for passo in passos_dict}  # Inicialmente, todos os passos são considerados válidos

    for passo, conteudo in passos_dict.items():
        if "(Hipótese)" in conteudo or "(Hip-PC)" in conteudo or "(Hip-RAA)" in conteudo:
            continue  # Hipóteses já são validadas separadamente

        # Extrair os passos referenciados e a regra aplicada
        passos_referenciados = []
        regra = ""
        if "MP(" in conteudo:
            passos_referenciados = re.findall(r'MP\((\d+),\s*(\d+)\)', conteudo)[0]
            regra = "MP"
        elif "CJ(" in conteudo:
            passos_referenciados = re.findall(r'CJ\((\d+),\s*(\d+)\)', conteudo)[0]
            regra = "CJ"
        elif "SD(" in conteudo:
            passos_referenciados = re.findall(r'SD\((\d+),\s*(\d+)\)', conteudo)[0]
            regra = "SD"
        elif "SH(" in conteudo:
            passos_referenciados = re.findall(r'SH\((\d+),\s*(\d+)\)', conteudo)[0]
            regra = "SH"
        elif "DN(" in conteudo:
            passos_referenciados = re.findall(r'DN\((\d+)\)', conteudo)[0]
            regra = "DN"
        elif "AD(" in conteudo:
            passos_referenciados = re.findall(r'AD\((\d+)\)', conteudo)[0]
            regra = "AD"
        elif "vE(" in conteudo:
            passos_referenciados = re.findall(r'vE\((\d+),\s*(\d+),\s*(\d+)\)', conteudo)[0]
            regra = "vE"
        elif "↔I(" in conteudo:
            passos_referenciados = re.findall(r'↔I\((\d+),\s*(\d+)\)', conteudo)[0]
            regra = "↔I"
        elif "↔E(" in conteudo:
            passos_referenciados = re.findall(r'↔E\((\d+)\)', conteudo)[0]
            regra = "↔E"
        elif "PC(" in conteudo:
            passos_referenciados = re.findall(r'PC\((\d+)-(\d+)\)', conteudo)[0]
            regra = "PC"
        elif "RAA(" in conteudo:
            passos_referenciados = re.findall(r'RAA\((\d+)-(\d+)\)', conteudo)[0]
            regra = "RAA"
        elif "COM(" in conteudo:
            passos_referenciados = re.findall(r'COM\((\d+)\)', conteudo)[0]
            regra = "COM"
        elif "DMOR(" in conteudo:
            passos_referenciados = re.findall(r'DMOR\((\d+)\)', conteudo)[0]
            regra = "DMOR"
        elif "COND(" in conteudo:
            passos_referenciados = re.findall(r'COND\((\d+)\)', conteudo)[0]
            regra = "COND"

        # Verificar se os passos referenciados são válidos
        if passos_referenciados:
            valido = True
            for ref in passos_referenciados:
                if not validade_passos.get(ref, False):
                    valido = False
                    feedback += f"Erro no passo {passo}: O passo referenciado {ref} é inválido.\n"
                    break

            # Se os passos referenciados forem válidos, validar a regra específica
            if valido:
                if regra == "MP":
                    valido, feedback = validar_modus_ponens(passo, conteudo, passos_dict, feedback)
                elif regra == "CJ":
                    valido, feedback = validar_conjuncao(passo, conteudo, passos_dict, feedback)
                elif regra == "SD":
                    valido, feedback = validar_silogismo_disjuntivo(passo, conteudo, passos_dict, feedback)
                elif regra == "SH":
                    valido, feedback = validar_silogismo_hipotetico(passo, conteudo, passos_dict, feedback)
                elif regra == "DN":
                    valido, feedback = validar_dupla_negacao(passo, conteudo, passos_dict, feedback)
                elif regra == "AD":
                    valido, feedback = validar_adicao(passo, conteudo, passos_dict, feedback)
                elif regra == "vE":
                    valido, feedback = validar_eliminacao_disjuncao(passo, conteudo, passos_dict, feedback)
                elif regra == "↔I":
                    valido, feedback = validar_introducao_equivalencia(passo, conteudo, passos_dict, feedback)
                elif regra == "↔E":
                    valido, feedback = validar_eliminacao_equivalencia(passo, conteudo, passos_dict, feedback)
                elif regra == "PC":
                    valido, feedback = validar_prova_condicional(passo, conteudo, passos_dict, feedback)
                elif regra == "RAA":
                    valido, feedback = validar_reducao_ao_absurdo(passo, conteudo, passos_dict, feedback, conclusao)
                elif regra == "COM":
                    valido, feedback = validar_comutatividade(passo, conteudo, passos_dict, feedback)
                elif regra == "DMOR":
                    valido, feedback = validar_de_morgan(passo, conteudo, passos_dict, feedback)
                elif regra == "COND":
                    valido, feedback = validar_condicional(passo, conteudo, passos_dict, feedback)

            # Adicionar a relação ao grafo
            relacoes.append((passos_referenciados, passo, regra, valido))
            validade_passos[passo] = valido

    return relacoes, feedback
# Função para validar a dedução
def validar(premissas_e_conclusao, passos):
    feedback = "Processando a dedução...\n"
    valido = True
    caminho_grafo = None

    try:
        # Conversão e processamento das premissas e passos
        premissas_e_conclusao = converter_latex_para_legivel(premissas_e_conclusao)
        passos = converter_latex_para_legivel(passos)

        # Separar premissas e conclusão
        if "⊢" not in premissas_e_conclusao:
            raise ValueError("A entrada deve conter o símbolo '⊢' para separar premissas e conclusão.")
        
        premissas, conclusao = premissas_e_conclusao.split("⊢")
        premissas = [p.strip() for p in premissas.split(",")]
        conclusao = [c.strip() for c in conclusao.split(",")]

        feedback += f"Premissas: {premissas}\n"
        feedback += f"Conclusão: {conclusao}\n"

        # Processar passos da dedução
        passos_dict = {}
        for linha in passos.splitlines():
            if linha.strip():
                if "." not in linha:
                    raise ValueError("Cada linha de passos deve conter um ponto (.) para separar o número do passo do conteúdo.")
                
                numero, conteudo = linha.split(".", 1)
                if not numero.strip().isdigit():
                    raise ValueError(f"O número do passo '{numero}' não é válido.")
                passos_dict[numero.strip()] = conteudo.strip()

        feedback += "Passos detectados:\n"
        for passo, conteudo in passos_dict.items():
            feedback += f"{passo}: {conteudo}\n"

        # Validar hipóteses
        valido_hipoteses, feedback = validar_hipoteses(premissas, passos_dict, feedback, conclusao)
        valido = valido and valido_hipoteses

        # Validar as regras de dedução
        try:
            validade_passos, feedback = validar_regras(premissas, passos_dict, conclusao, feedback)
            valido = all(validade_passos.values())
        except DeducaoInvalidaError as e:
            valido = False
            feedback += str(e) + "\n"

        # Verificar a conclusão
        conclusoes_deduzidas = [extrair_conclusao(valor) for valor in passos_dict.values()]
        for c in conclusao:
            if not any(proposicoes_iguais(c, deduzida) for deduzida in conclusoes_deduzidas):
                feedback += (f"Erro: A conclusão '{c}' não foi encontrada entre os passos deduzidos.\n"
                            "Verifique se ela foi construída corretamente com as premissas e as regras aplicadas.\n"
                )
                valido = False
            else:
                feedback += f"Conclusão '{c}' foi deduzida corretamente.\n"

        # Coletar as relações entre os passos
        relacoes, feedback = coletar_relacoes(passos_dict, feedback, conclusao)

        # Gerar o grafo da dedução
        try:
            caminho_grafo = gerar_grafo(passos_dict, relacoes, premissas, validade_passos, conclusao)
            feedback += "Grafo gerado com sucesso.\n"
    
        except FileNotFoundError:
            feedback += "Erro: Graphviz não está instalado. O grafo não foi gerado.\n"
        
        except Exception as e:
            return False, f"Erro durante a validação: {e}", None
        
    except Exception as e:
        return False, f"Erro durante o processamento: {e}", None

    return valido, feedback, caminho_grafo

def validar_hipoteses(premissas, passos_dict, feedback, conclusao):
    """
    Verifica se os passos marcados como hipóteses estão corretos (i.e., presentes nas premissas ou são o antecedente da conclusão no caso de Hip-PC).
    """
    valido = True
    for passo, conteudo in passos_dict.items():
        if "(Hipótese)" in conteudo:
            proposicao = extrair_conclusao(conteudo.replace("(Hipótese)", "").strip())
            if not any(proposicoes_iguais(proposicao, premissa) for premissa in premissas):
                feedback += f"Erro no passo {passo}: A hipótese '{proposicao}' não corresponde a nenhuma premissa.\n"
                valido = False
        elif "(Hip-RAA)" in conteudo:
            proposicao = extrair_conclusao(conteudo.replace("(Hip-RAA)", "").strip())
            # A suposição Hip-RAA deve ser a negação da conclusão
            if not any(proposicoes_iguais(proposicao, f"¬{c}") or proposicoes_iguais(f"¬{proposicao}", c) for c in conclusao):
                feedback += f"Erro no passo {passo}: A suposição Hip-RAA '{proposicao}' não é a negação da conclusão.\n"
                valido = False
        elif "(Hip-PC)" in conteudo:
            proposicao = extrair_conclusao(conteudo.replace("(Hip-PC)", "").strip())
            # A suposição Hip-PC deve ser o antecedente da conclusão da prova condicional
            if not any(extrair_antecedente_consequente(c)[0] == proposicao for c in conclusao if "→" in c):
                feedback += f"Erro no passo {passo}: A suposição Hip-PC '{proposicao}' não é o antecedente da conclusão.\n"
                valido = False
    return valido, feedback


def passos_referenciados_validos(passo, passos_referenciados, validade_passos, regra, feedback):
    """
    Verifica se todos os passos referenciados são válidos.
    Retorna True se todos forem válidos, False caso contrário.
    """
    for ref in passos_referenciados:
        if not validade_passos.get(ref, False):
            feedback += f"Erro no passo {passo} ({regra}): O passo referenciado {ref} é inválido.\n"
            return False
    return True

# Função para validar as regras de dedução
def validar_regras(premissas, passos_dict, conclusao, feedback):
    """
    Valida a aplicação das regras de dedução (MP, SP, CJ, etc.).
    """
    validade_passos = {passo: True for passo in passos_dict}  # Inicialmente, todos os passos são considerados válidos

    for passo, conteudo in passos_dict.items():
        if "(Hipótese)" in conteudo:
            continue  # Hipóteses já são validadas separadamente

        # Verifica se há referências a passos anteriores
        passos_referenciados = []
        regra = ""
        if "MP(" in conteudo:
            passos_referenciados = re.findall(r'MP\((\d+),\s*(\d+)\)', conteudo)[0]
            regra = "MP"
        elif "CJ(" in conteudo:
            passos_referenciados = re.findall(r'CJ\((\d+),\s*(\d+)\)', conteudo)[0]
            regra = "CJ"
        elif "SD(" in conteudo:
            passos_referenciados = re.findall(r'SD\((\d+),\s*(\d+)\)', conteudo)[0]
            regra = "SD"
        elif "SH(" in conteudo:
            passos_referenciados = re.findall(r'SH\((\d+),\s*(\d+)\)', conteudo)[0]
            regra = "SH"
        elif "DN(" in conteudo:
            passos_referenciados = re.findall(r'DN\((\d+)\)', conteudo)[0]
            regra = "DN"
        elif "AD(" in conteudo:
            passos_referenciados = re.findall(r'AD\((\d+)\)', conteudo)[0]
            regra = "AD"
        elif "vE(" in conteudo:
            passos_referenciados = re.findall(r'vE\((\d+),\s*(\d+),\s*(\d+)\)', conteudo)[0]
            regra = "vE"
        elif "↔I(" in conteudo:
            passos_referenciados = re.findall(r'↔I\((\d+),\s*(\d+)\)', conteudo)[0]
            regra = "↔I"
        elif "↔E(" in conteudo:
            passos_referenciados = re.findall(r'↔E\((\d+)\)', conteudo)[0]
            regra = "↔E"
        elif "PC(" in conteudo:
            passos_referenciados = re.findall(r'PC\((\d+)-(\d+)\)', conteudo)[0]
            regra = "PC"
        elif "RAA(" in conteudo:
            passos_referenciados = re.findall(r'RAA\((\d+)-(\d+)\)', conteudo)[0]
            regra = "RAA"
        elif "COM(" in conteudo:
            passos_referenciados = re.findall(r'COM\((\d+)\)', conteudo)[0]
            regra = "COM"
        elif "DMOR(" in conteudo:
            passos_referenciados = re.findall(r'DMOR\((\d+)\)', conteudo)[0]
            regra = "DMOR"
        elif "COND(" in conteudo:
            passos_referenciados = re.findall(r'COND\((\d+)\)', conteudo)[0]
            regra = "COND"

        # Verifica se os passos referenciados são válidos
        if passos_referenciados:
            if not passos_referenciados_validos(passo, passos_referenciados, validade_passos, regra, feedback):
                validade_passos[passo] = False
                continue  # Pula a validação da regra se os passos referenciados forem inválidos

        # Valida a regra específica
        if "MP(" in conteudo:
            valido_mp, feedback = validar_modus_ponens(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_mp and validade_passos[passo]
        elif "CJ(" in conteudo:
            valido_cj, feedback = validar_conjuncao(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_cj and validade_passos[passo]
        elif "SD(" in conteudo:
            valido_sd, feedback = validar_silogismo_disjuntivo(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_sd and validade_passos[passo]
        elif "SH(" in conteudo:
            valido_sh, feedback = validar_silogismo_hipotetico(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_sh and validade_passos[passo]
        elif "DN(" in conteudo:
            valido_dn, feedback = validar_dupla_negacao(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_dn and validade_passos[passo]
        elif "AD(" in conteudo:
            valido_ad, feedback = validar_adicao(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_ad and validade_passos[passo]
        elif "vE(" in conteudo:
            valido_ve, feedback = validar_eliminacao_disjuncao(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_ve and validade_passos[passo]
        elif "↔I(" in conteudo:
            valido_ie, feedback = validar_introducao_equivalencia(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_ie and validade_passos[passo]
        elif "↔E(" in conteudo:
            valido_ee, feedback = validar_eliminacao_equivalencia(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_ee and validade_passos[passo]
        elif "PC(" in conteudo:
            valido_pc, feedback = validar_prova_condicional(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_pc and validade_passos[passo]
        elif "RAA(" in conteudo:
            valido_raa, feedback = validar_reducao_ao_absurdo(passo, conteudo, passos_dict, feedback, conclusao)
            validade_passos[passo] = valido_raa and validade_passos[passo]
        elif "COM(" in conteudo:
            valido_com, feedback = validar_comutatividade(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_com and validade_passos[passo]
        elif "DMOR(" in conteudo:
            valido_dmor, feedback = validar_de_morgan(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_dmor and validade_passos[passo]
        elif "COND(" in conteudo:
            valido_cond, feedback = validar_condicional(passo, conteudo, passos_dict, feedback)
            validade_passos[passo] = valido_cond and validade_passos[passo]

    return validade_passos, feedback

def validar_modus_ponens(passo, conteudo, passos_dict, feedback):
    try:
        match = re.search(r'MP\((\d+),\s*(\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "MP", "Modus Ponens requer justificativas (ex.: MP(1,2))")

        passo_condicional, passo_antecedente = match.groups()
        
        if passo_condicional not in passos_dict or passo_antecedente not in passos_dict:
            raise DeducaoInvalidaError(passo, "MP", f"Referências {passo_condicional} ou {passo_antecedente} não encontradas")

        condicional = extrair_conclusao(passos_dict[passo_condicional])
        antecedente = extrair_conclusao(passos_dict[passo_antecedente])

        if "→" not in condicional:
            raise DeducaoInvalidaError(passo, "MP", f"Referência {passo_condicional} não é uma condicional")

        partes = condicional.split("→")
        if not proposicoes_iguais(antecedente, partes[0]):
            raise DeducaoInvalidaError(passo, "MP", f"Antecedente '{antecedente}' não corresponde a '{partes[0]}'")

        conclusao_deduzida = extrair_conclusao(conteudo)
        if not proposicoes_iguais(conclusao_deduzida, partes[1]):
            raise DeducaoInvalidaError(passo, "MP", f"Conclusão esperada '{partes[1]}', mas encontrada '{conclusao_deduzida}'")

        feedback += f"Passo {passo} validado com sucesso usando Modus Ponens.\n"
        return True, feedback

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback

def validar_simplificacao(passo, conteudo, passos_dict, feedback):
    try:
        match = re.search(r'SP\((\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "SP", "Simplificação requer uma justificativa (ex.: SP(1))")

        passo_conjuncao = match.group(1)
        if passo_conjuncao not in passos_dict:
            raise DeducaoInvalidaError(passo, "SP", f"Passo {passo_conjuncao} não encontrado")

        conjuncao = extrair_conclusao(passos_dict[passo_conjuncao])
        if "∧" not in conjuncao:
            raise DeducaoInvalidaError(passo, "SP", f"Referência {passo_conjuncao} não é uma conjunção")

        partes_conjuncao = conjuncao.split("∧")
        conclusao_deduzida = extrair_conclusao(conteudo)
        
        if any(proposicoes_iguais(conclusao_deduzida, parte) for parte in partes_conjuncao):
            feedback += f"Passo {passo} validado com sucesso usando Simplificação.\n"
            return True, feedback
        else:
            raise DeducaoInvalidaError(passo, "SP", f"O conteúdo '{conclusao_deduzida}' não é uma simplificação válida da conjunção '{conjuncao}'")

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback

def validar_conjuncao(passo, conteudo, passos_dict, feedback):
    try:
        # Verifica se a justificativa da conjunção está no formato correto (CJ(n1, n2))
        match = re.search(r'CJ\((\d+),\s*(\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "CJ", "Conjunção requer duas justificativas (ex.: CJ(1,2))")

        passo_1, passo_2 = match.groups()
        
        # Verifica se os passos referenciados existem
        if passo_1 not in passos_dict or passo_2 not in passos_dict:
            raise DeducaoInvalidaError(passo, "CJ", f"Referências {passo_1} ou {passo_2} não encontradas")

        # Extrai as proposições dos passos referenciados
        proposicao1 = extrair_conclusao(passos_dict[passo_1])
        proposicao2 = extrair_conclusao(passos_dict[passo_2])

        # Extrai a conclusão deduzida no passo atual
        conclusao_deduzida = extrair_conclusao(conteudo)

        # Verifica se a conclusão deduzida é uma conjunção válida das proposições
        # A ordem das proposições pode variar (p ∧ q ou q ∧ p)
        if not (proposicoes_iguais(conclusao_deduzida, f"{proposicao1}∧{proposicao2}") or
                proposicoes_iguais(conclusao_deduzida, f"{proposicao2}∧{proposicao1}")):
            raise DeducaoInvalidaError(passo, "CJ", f"A conclusão '{conclusao_deduzida}' não corresponde à união de '{proposicao1}' e '{proposicao2}'")

        # Se tudo estiver correto, retorna True e o feedback
        feedback += f"Passo {passo} validado com sucesso usando Conjunção.\n"
        return True, feedback

    except DeducaoInvalidaError as e:
        # Se houver erro, retorna False e o feedback de erro
        feedback += str(e) + "\n"
        return False, feedback
    
def extrair_antecedente_consequente(proposicao):
    """Extrai o antecedente e o consequente de uma proposição condicional,
    lidando com proposições aninhadas e parênteses."""

    if "→" not in proposicao:
        return None, None

    # Encontra o índice do operador condicional principal
    nivel_parenteses = 0
    indice_condicional = -1
    for i, char in enumerate(proposicao):
        if char == "(":
            nivel_parenteses += 1
        elif char == ")":
            nivel_parenteses -= 1
        elif char == "→" and nivel_parenteses == 0:
            indice_condicional = i
            break

    if indice_condicional == -1:  # Não encontrou um condicional válido
        return None, None

    antecedente = proposicao[:indice_condicional].strip()
    consequente = proposicao[indice_condicional + 1:].strip()
    return antecedente, consequente


def validar_modus_tollens(passo, conteudo, passos_dict, feedback):
    try:
        match = re.search(r'MT\((\d+),\s*(\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "MT", "Modus Tollens requer justificativas (ex.: MT(1,2))")
        
        passo_condicional, passo_negacao = match.groups()

        if passo_condicional not in passos_dict or passo_negacao not in passos_dict:
            raise DeducaoInvalidaError(passo, "MT", f"Referências {passo_condicional} ou {passo_negacao} não encontradas")

        condicional = extrair_conclusao(passos_dict[passo_condicional])
        negacao_consequente_fornecida = extrair_conclusao(passos_dict[passo_negacao])
        conclusao_deduzida = extrair_conclusao(conteudo)

        antecedente, consequente = extrair_antecedente_consequente(condicional)

        if antecedente is None or consequente is None:
            raise DeducaoInvalidaError(passo, "MT", f"A proposição no passo {passo_condicional} não é uma condicional válida")


        # Verifica se a negação corresponde ao consequente ou se a segunda premissa
        # é equivalente à negação do consequente (ex: t e ¬¬t)
        negacao_consequente = f"¬{consequente}"
        if not (proposicoes_iguais(negacao_consequente_fornecida, negacao_consequente) or \
                proposicoes_iguais(f"¬{negacao_consequente_fornecida}", consequente)):
            raise DeducaoInvalidaError(passo, "MT", f"A negação '{negacao_consequente_fornecida}' não corresponde ao esperado '¬{consequente}'")

        # Verifica se a conclusão é a negação do antecedente
        if not proposicoes_iguais(conclusao_deduzida, f"¬{antecedente}"):
            raise DeducaoInvalidaError(passo, "MT", f"Conclusão esperada '¬{antecedente}', mas encontrada '{conclusao_deduzida}'")


        feedback += f"Passo {passo} validado com sucesso usando Modus Tollens.\n"
        return True, feedback

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback


def validar_silogismo_disjuntivo(passo, conteudo, passos_dict, feedback):
    try:
        # Extrair os passos usados na regra SD
        match = re.search(r'SD\((\d+),\s*(\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "SD", "Silogismo Disjuntivo requer duas justificativas (ex.: SD(1,2))")
        
        passo_disjuncao, passo_negacao = match.groups()

        # Validar a existência dos passos referenciados
        if passo_disjuncao not in passos_dict or passo_negacao not in passos_dict:
            raise DeducaoInvalidaError(passo, "SD", f"Referências {passo_disjuncao} ou {passo_negacao} não encontradas")

        # Extrair as proposições dos passos referenciados
        disjuncao = extrair_conclusao(passos_dict[passo_disjuncao])
        negacao = extrair_conclusao(passos_dict[passo_negacao])

        # Validar se o passo referenciado contém uma disjunção
        if "v" not in disjuncao:
            raise DeducaoInvalidaError(passo, "SD", f"Referência {passo_disjuncao} não é uma disjunção")

        # Validar se a segunda referência é a negação de uma das proposições
        partes_disjuncao = [p.strip() for p in disjuncao.split("v")]
        if not any(proposicoes_iguais(negacao, f"¬{parte}") for parte in partes_disjuncao):
            raise DeducaoInvalidaError(passo, "SD", f"A negação '{negacao}' não corresponde a nenhuma das proposições na disjunção '{disjuncao}'")

        # Validar a conclusão deduzida
        conclusao_deduzida = extrair_conclusao(conteudo)
        proposicao_deduzida = next(parte for parte in partes_disjuncao if not proposicoes_iguais(negacao, f"¬{parte}"))

        if not proposicoes_iguais(conclusao_deduzida, proposicao_deduzida):
            raise DeducaoInvalidaError(passo, "SD", f"A conclusão '{conclusao_deduzida}' não corresponde à proposição esperada '{proposicao_deduzida}'")

        feedback += f"Passo {passo} validado com sucesso usando Silogismo Disjuntivo.\n"
        return True, feedback

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback
    
def validar_silogismo_hipotetico(passo, conteudo, passos_dict, feedback):
    try:
        match = re.search(r'SH\((\d+),\s*(\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "SH", "Silogismo Hipotético requer duas justificativas (ex.: SH(1,2))")

        passo_1, passo_2 = match.groups()
        
        if passo_1 not in passos_dict or passo_2 not in passos_dict:
            raise DeducaoInvalidaError(passo, "SH", f"Referências {passo_1} ou {passo_2} não encontradas")

        condicional_1 = extrair_conclusao(passos_dict[passo_1])
        condicional_2 = extrair_conclusao(passos_dict[passo_2])

        if "→" not in condicional_1 or "→" not in condicional_2:
            raise DeducaoInvalidaError(passo, "SH", "Ambas as referências devem ser condicionais")

        antecedente_1, consequente_1 = [p.strip() for p in condicional_1.split("→")]
        antecedente_2, consequente_2 = [p.strip() for p in condicional_2.split("→")]

        if not proposicoes_iguais(consequente_1, antecedente_2):
            raise DeducaoInvalidaError(passo, "SH", f"O consequente de '{condicional_1}' não corresponde ao antecedente de '{condicional_2}'")

        conclusao_esperada = f"{antecedente_1} → {consequente_2}"
        conclusao_deduzida = extrair_conclusao(conteudo)

        if not proposicoes_iguais(conclusao_deduzida, conclusao_esperada):
            raise DeducaoInvalidaError(passo, "SH", f"A conclusão esperada era '{conclusao_esperada}', mas foi encontrada '{conclusao_deduzida}'")

        feedback += f"Passo {passo} validado com sucesso usando Silogismo Hipotético.\n"
        return True, feedback

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback



def validar_dupla_negacao(passo, conteudo, passos_dict, feedback):
    try:
        match = re.search(r'DN\((\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "DN", "Dupla Negação requer uma justificativa (ex.: DN(1))")

        passo_justificado2 = match.group(1)
        if passo_justificado2 not in passos_dict:
            raise DeducaoInvalidaError(passo, "DN", f"Passo {passo_justificado2} não encontrado")

        proposicao = extrair_conclusao(passos_dict[passo_justificado2])
        conclusao_deduzida = extrair_conclusao(conteudo)

        if proposicoes_iguais(proposicao, f"¬¬{conclusao_deduzida}") or proposicoes_iguais(f"¬¬{proposicao}", conclusao_deduzida):
            feedback += f"Passo {passo} validado com sucesso usando Dupla Negação.\n"
            return True, feedback
        else:
            raise DeducaoInvalidaError(passo, "DN", f"A conclusão '{conclusao_deduzida}' não corresponde à dupla negação de '{proposicao}'")

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback
    

def validar_adicao(passo, conteudo, passos_dict, feedback):
    try:
        match = re.search(r'AD\((\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "AD", "Adição requer uma justificativa (ex.: AD(1))")
        
        passo_justificado3 = match.group(1)  # Corrigido para usar group(1)
        if passo_justificado3 not in passos_dict:
            raise DeducaoInvalidaError(passo, "AD", f"Passo {passo_justificado3} não encontrado")
        
        proposicao = extrair_conclusao(passos_dict[passo_justificado3])
        conclusao_deduzida = extrair_conclusao(conteudo)
        
        if proposicao not in conclusao_deduzida or "v" not in conclusao_deduzida:
            raise DeducaoInvalidaError(passo, "AD", f"A conclusão '{conclusao_deduzida}' não é uma adição válida de '{proposicao}'")
    
        feedback += f"Passo {passo} validado com sucesso usando Adição.\n"
        return True, feedback
    
    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback

def validar_eliminacao_disjuncao(passo, conteudo, passos_dict, feedback):
    try:
        match = re.search(r'vE\((\d+),\s*(\d+),\s*(\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "vE", "Eliminação da Disjunção requer três justificativas (ex.: vE(1,2,3))")

        passo_disjuncao, passo_implicacao1, passo_implicacao2 = match.groups()
        
        if passo_disjuncao not in passos_dict or passo_implicacao1 not in passos_dict or passo_implicacao2 not in passos_dict:
            raise DeducaoInvalidaError(passo, "vE", f"Referências {passo_disjuncao}, {passo_implicacao1} ou {passo_implicacao2} não encontradas")

        disjuncao = extrair_conclusao(passos_dict[passo_disjuncao])
        implicacao1 = extrair_conclusao(passos_dict[passo_implicacao1])
        implicacao2 = extrair_conclusao(passos_dict[passo_implicacao2])

        # Verifique se a disjunção contém exatamente duas partes
        partes_disjuncao = [p.strip() for p in disjuncao.split("v")]
        if len(partes_disjuncao) != 2:
            raise DeducaoInvalidaError(passo, "vE", "A disjunção deve conter exatamente duas proposições")

        # Lógica para dividir implicações
        def encontrar_divisao(implicacao):
            contador_parenteses = 0
            for i, char in enumerate(implicacao):
                if char == '(':
                    contador_parenteses += 1
                elif char == ')':
                    contador_parenteses -= 1
                elif char == '→' and contador_parenteses == 0:
                    return i
            return -1

        # Divida as implicações
        indice_divisao1 = encontrar_divisao(implicacao1)
        antecedente1 = implicacao1[:indice_divisao1].strip()
        consequente1 = implicacao1[indice_divisao1 + 1:].strip()

        indice_divisao2 = encontrar_divisao(implicacao2)
        antecedente2 = implicacao2[:indice_divisao2].strip()
        consequente2 = implicacao2[indice_divisao2 + 1:].strip()

        # Verifique se cada parte da disjunção implica no mesmo consequente
        if not ((proposicoes_iguais(partes_disjuncao[0], antecedente1) and proposicoes_iguais(consequente1, consequente2)) or
                (proposicoes_iguais(partes_disjuncao[1], antecedente2) and proposicoes_iguais(consequente1, consequente2))):
            raise DeducaoInvalidaError(passo, "vE", "As partes da disjunção não implicam no mesmo consequente")

        # Verifique se a conclusão deduzida é o consequente comum
        conclusao_deduzida = extrair_conclusao(conteudo)
        if not proposicoes_iguais(conclusao_deduzida, consequente1):
            raise DeducaoInvalidaError(passo, "vE", f"A conclusão esperada era '{consequente1}', mas foi encontrada '{conclusao_deduzida}'")

        feedback += f"Passo {passo} validado com sucesso usando Eliminação da Disjunção.\n"
        return True, feedback

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback
    
def validar_introducao_equivalencia(passo, conteudo, passos_dict, feedback):
    try:
        match = re.search(r'↔I\((\d+),\s*(\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "↔I", "Introdução da Equivalência requer duas justificativas (ex.: ↔I(1,2))")

        passo_implicacao1, passo_implicacao2 = match.groups()
        
        if passo_implicacao1 not in passos_dict or passo_implicacao2 not in passos_dict:
            raise DeducaoInvalidaError(passo, "↔I", f"Referências {passo_implicacao1} ou {passo_implicacao2} não encontradas")

        implicacao1 = extrair_conclusao(passos_dict[passo_implicacao1])
        implicacao2 = extrair_conclusao(passos_dict[passo_implicacao2])

        # Divida as implicações
        indice_divisao1 = implicacao1.find("→")
        antecedente1 = implicacao1[:indice_divisao1].strip()
        consequente1 = implicacao1[indice_divisao1 + 1:].strip()

        indice_divisao2 = implicacao2.find("→")
        antecedente2 = implicacao2[:indice_divisao2].strip()
        consequente2 = implicacao2[indice_divisao2 + 1:].strip()

        # Verifique se as implicações são inversas
        if not (proposicoes_iguais(antecedente1, consequente2) and proposicoes_iguais(consequente1, antecedente2)):
            raise DeducaoInvalidaError(passo, "↔I", "As implicações não são inversas")

        # Verifique se a conclusão deduzida é a equivalência
        conclusao_deduzida = extrair_conclusao(conteudo)
        equivalencia_esperada = f"{antecedente1} ↔ {consequente1}"
        if not proposicoes_iguais(conclusao_deduzida, equivalencia_esperada):
            raise DeducaoInvalidaError(passo, "↔I", f"A conclusão esperada era '{equivalencia_esperada}', mas foi encontrada '{conclusao_deduzida}'")

        feedback += f"Passo {passo} validado com sucesso usando Introdução da Equivalência.\n"
        return True, feedback

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback

    
def validar_eliminacao_equivalencia(passo, conteudo, passos_dict, feedback):
    try:
        match = re.search(r'↔E\((\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "↔E", "Eliminação da Equivalência requer uma justificativa (ex.: ↔E(1))")

        passo_equivalencia = match.group(1)
        
        if passo_equivalencia not in passos_dict:
            raise DeducaoInvalidaError(passo, "↔E", f"Referência {passo_equivalencia} não encontrada")

        equivalencia = extrair_conclusao(passos_dict[passo_equivalencia])

        # Divida a equivalência
        partes_equivalencia = [p.strip() for p in equivalencia.split("↔")]
        if len(partes_equivalencia) != 2:
            raise DeducaoInvalidaError(passo, "↔E", "A equivalência deve conter exatamente duas proposições")

        antecedente, consequente = partes_equivalencia

        # Verifique se a conclusão deduzida é uma das implicações
        conclusao_deduzida = extrair_conclusao(conteudo)
        implicacao1 = f"{antecedente} → {consequente}"
        implicacao2 = f"{consequente} → {antecedente}"

        if not (proposicoes_iguais(conclusao_deduzida, implicacao1) or proposicoes_iguais(conclusao_deduzida, implicacao2)):
            raise DeducaoInvalidaError(passo, "↔E", f"A conclusão esperada era '{implicacao1}' ou '{implicacao2}', mas foi encontrada '{conclusao_deduzida}'")

        feedback += f"Passo {passo} validado com sucesso usando Eliminação da Equivalência.\n"
        return True, feedback

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback


def validar_prova_condicional(passo, conteudo, passos_dict, feedback):
    try:
        # Verifica se a sintaxe da regra PC está correta
        match = re.search(r'PC\((\d+)-(\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "PC", "Prova Condicional requer um intervalo de passos (ex.: PC(3-5))")

        passo_inicio, passo_fim = match.groups()
        
        # Verifica se os passos referenciados existem
        if passo_inicio not in passos_dict or passo_fim not in passos_dict:
            raise DeducaoInvalidaError(passo, "PC", f"Referências {passo_inicio} ou {passo_fim} não encontradas")

        # Extrai as proposições relevantes
        antecedente = extrair_conclusao(passos_dict[passo_inicio])
        consequente = extrair_conclusao(passos_dict[passo_fim])
        conclusao_deduzida = extrair_conclusao(conteudo)

        # Verifica se o passo inicial é uma hipótese adicional marcada como Hip-PC
        if "(Hip-PC)" not in passos_dict[passo_inicio]:
            raise DeducaoInvalidaError(
                passo, "PC", f"O passo inicial '{passo_inicio}' não é uma hipótese adicional marcada como Hip-PC."
            )

        # Verifica se a conclusão deduzida é do tipo antecedente → consequente
        if not proposicoes_iguais(conclusao_deduzida, f"{antecedente}→{consequente}"):
            raise DeducaoInvalidaError(
                passo, "PC", f"A conclusão '{conclusao_deduzida}' não corresponde a '{antecedente}→{consequente}'"
            )

        feedback += f"Passo {passo} validado com sucesso usando Prova Condicional (PC).\n"
        return True, feedback

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback

def eh_contradicao(proposicao):
    """
    Verifica se a proposição é uma contradição lógica do tipo A ∧ ¬A.
    """
    proposicao = proposicao.replace(" ", "")
    partes = proposicao.split("∧")
    if len(partes) != 2:
        return False
    parte1, parte2 = partes
    return proposicoes_iguais(parte1, f"¬{parte2}") or proposicoes_iguais(parte2, f"¬{parte1}")

def validar_reducao_ao_absurdo(passo, conteudo, passos_dict, feedback, conclusao):
    try:
        match = re.search(r'RAA\((\d+)-(\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "RAA", "Formato inválido para RAA (ex.: RAA(3-5))")
        
        passo_inicio, passo_fim = match.groups()
        
        if passo_inicio not in passos_dict or passo_fim not in passos_dict:
            raise DeducaoInvalidaError(passo, "RAA", f"Referências {passo_inicio} ou {passo_fim} não encontradas")

        if "(Hip-RAA)" not in passos_dict[passo_inicio]:
            raise DeducaoInvalidaError(passo, "RAA", f"O passo inicial '{passo_inicio}' não é marcado como Hip-RAA.")

        suposicao = extrair_conclusao(passos_dict[passo_inicio])
        conclusao_contradicao = extrair_conclusao(passos_dict[passo_fim])

        if not eh_contradicao(conclusao_contradicao):
            raise DeducaoInvalidaError(passo, "RAA", f"A conclusão '{conclusao_contradicao}' não é uma contradição válida")
    
        if not (proposicoes_iguais(suposicao, f"¬{conclusao[0]}") or proposicoes_iguais(f"¬{suposicao}", conclusao[0])):
            raise DeducaoInvalidaError(passo, "RAA", f"A suposição '{suposicao}' não é a negação da conclusão '{conclusao[0]}'")

        # Verifica se a conclusão derivada corresponde à conclusão esperada
        conclusao_deduzida = extrair_conclusao(conteudo)
        if not proposicoes_iguais(conclusao_deduzida, conclusao[0]):
            raise DeducaoInvalidaError(passo, "RAA", f"A conclusão derivada '{conclusao_deduzida}' não corresponde à conclusão esperada '{conclusao[0]}'")

        feedback += f"Passo {passo} validado com sucesso usando Redução ao Absurdo.\n"
        return True, feedback

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback

def validar_comutatividade(passo, conteudo, passos_dict, feedback):
    try:
        # Procurar pela justificativa de comutatividade no formato COM(n)
        match = re.search(r'COM\((\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "COM", "Comutatividade requer justificativa no formato COM(número do passo)")
        
        passo_origem = match.group(1)
        if passo_origem not in passos_dict:
            raise DeducaoInvalidaError(passo, "COM", f"Referência ao passo {passo_origem} não encontrada")
        
        origem = extrair_conclusao(passos_dict[passo_origem])
        resultado = extrair_conclusao(conteudo)

        # Dividir as proposições em partes usando operadores lógicos
        partes_origem = re.split(r'([∧∨])', origem)
        partes_resultado = re.split(r'([∧∨])', resultado)

        # Verificar se as partes são equivalentes, considerando a comutatividade
        if len(partes_origem) != len(partes_resultado):
            raise DeducaoInvalidaError(passo, "COM", f"'{origem}' não é equivalente a '{resultado}' por comutatividade")

        # Comparar cada parte, permitindo troca de ordem
        for i in range(0, len(partes_origem), 2):
            if i+1 < len(partes_resultado) and not (
                proposicoes_iguais(partes_origem[i], partes_resultado[i]) or
                proposicoes_iguais(partes_origem[i], partes_resultado[i+1])
            ):
                raise DeducaoInvalidaError(passo, "COM", f"'{origem}' não é equivalente a '{resultado}' por comutatividade")
        
        feedback += f"Passo {passo} validado com sucesso usando Comutatividade.\n"
        return True, feedback
    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback

def simplificar(proposicao):
    """Simplifica negações duplas em uma proposição."""
    while "¬¬" in proposicao:
        proposicao = proposicao.replace("¬¬", "")
    return proposicao

def validar_de_morgan(passo, conteudo, passos_dict, feedback):
    """
    Valida a aplicação da regra De Morgan (DMOR).
    """
    try:
        match = re.search(r'DMOR\((\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "DMOR", "De Morgan requer uma justificativa (ex.: DMOR(1))")

        passo_referencia = match.group(1)
        if passo_referencia not in passos_dict:
            raise DeducaoInvalidaError(passo, "DMOR", f"Passo {passo_referencia} não encontrado")

        proposicao_original = extrair_conclusao(passos_dict[passo_referencia])
        conclusao_deduzida = extrair_conclusao(conteudo)

        if "¬(" in proposicao_original:
            proposicao_interna = proposicao_original[2:-1]  # Remove ¬( e )

            if "v" in proposicao_interna:
                partes = proposicao_interna.split("v")
                proposicao_esperada = f"¬{partes[0]}∧¬{partes[1]}"
            elif "∧" in proposicao_interna:
                partes = proposicao_interna.split("∧")
                proposicao_esperada = f"¬{partes[0]}∨¬{partes[1]}"
            else:
                raise DeducaoInvalidaError(passo, "DMOR", f"A proposição no passo {passo_referencia} não é uma conjunção ou disjunção negada válida para De Morgan.")

            proposicao_esperada = simplificar(proposicao_esperada)
            conclusao_deduzida = simplificar(conclusao_deduzida)

            if proposicoes_iguais(conclusao_deduzida, proposicao_esperada):
                feedback += f"Passo {passo} validado com sucesso usando De Morgan.\n"
                return True, feedback
            else:
                raise DeducaoInvalidaError(passo, "DMOR", f"A proposição '{conclusao_deduzida}' não corresponde à transformação esperada '{proposicao_esperada}'")
        else:  #  Caso em que não há negação principal, De Morgan não se aplica
            raise DeducaoInvalidaError(passo, "DMOR", "De Morgan requer uma negação antes da conjunção/disjunção.")

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback


def validar_condicional(passo, conteudo, passos_dict, feedback):
    """
    Valida a aplicação da regra Condicional (COND).
    """
    try:
        match = re.search(r'COND\((\d+)\)', conteudo)
        if not match:
            raise DeducaoInvalidaError(passo, "COND", "Condicional requer uma justificativa (ex.: COND(1))")

        passo_referencia = match.group(1)
        if passo_referencia not in passos_dict:
            raise DeducaoInvalidaError(passo, "COND", f"Passo {passo_referencia} não encontrado")

        proposicao_original = extrair_conclusao(passos_dict[passo_referencia])
        conclusao_deduzida = extrair_conclusao(conteudo)

        # Aplica a transformação condicional -> disjunção
        if "→" in proposicao_original:
            partes = proposicao_original.split("→")
            proposicao_esperada = f"¬{partes[0]}v{partes[1]}"
        elif "v" in proposicao_original:  # Inverso: disjunção -> condicional
            partes = proposicao_original.split("v")
            proposicao_esperada = f"¬{partes[0]}→{partes[1]}"
        else:
            raise DeducaoInvalidaError(passo, "COND", f"A proposição no passo {passo_referencia} não é uma condicional ou disjunção válida para a regra COND.")


        if proposicoes_iguais(conclusao_deduzida, proposicao_esperada):
            feedback += f"Passo {passo} validado com sucesso usando Condicional.\n"
            return True, feedback
        else:
            raise DeducaoInvalidaError(passo, "COND", f"A proposição '{conclusao_deduzida}' não corresponde à transformação esperada '{proposicao_esperada}'")

    except DeducaoInvalidaError as e:
        feedback += str(e) + "\n"
        return False, feedback

# Função para gerar o grafo da dedução
def gerar_grafo(passos_dict, relacoes, premissas, validade_passos, conclusao):
    """
    Gera o grafo da dedução, colorindo os nós com base na validade de cada passo.
    """
    if not os.path.exists("grafos"):
        os.makedirs("grafos")
    dot = Digraph(format="svg")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightgrey")
    dot.attr("edge", arrowsize="0.8", fontsize="10")

    # Valida as hipóteses (Hip-PC e Hip-RAA) usando a função validar_hipoteses
    valido_hipoteses, _ = validar_hipoteses(premissas, passos_dict, "", conclusao)

    # Adiciona os nós ao grafo com as cores corretas
    for numero, conteudo in passos_dict.items():
        # Verifica se o passo é uma hipótese (Hip-PC ou Hip-RAA)
        if "(Hip-PC)" in conteudo or "(Hip-RAA)" in conteudo:
            # Verifica se a hipótese está correta usando o resultado de validar_hipoteses
            if valido_hipoteses and validade_passos.get(numero, False):
                cor_no = "lightgreen"  # Verde se a hipótese estiver correta
            else:
                cor_no = "lightcoral"  # Vermelho se a hipótese estiver incorreta
        else:
            # Passos normais
            cor_no = "lightgreen" if validade_passos.get(numero, False) else "lightcoral"
        
        dot.node(numero, f"{numero}: {conteudo}", fillcolor=cor_no)

    # Adiciona as arestas ao grafo
    for refs, passo, regra, valido in relacoes:
        cor_aresta = "green" if valido else "red"
        label_aresta = f"{regra} {'(OK)' if valido else '(Erro)'}"
        for ref in refs:
            dot.edge(ref, passo, label=label_aresta, color=cor_aresta, fontcolor=cor_aresta)

    caminho_grafo = os.path.join("grafos", "grafo_deducao")
    dot.render(caminho_grafo, format="svg", cleanup=True)
    return f"{caminho_grafo}.svg"
    
