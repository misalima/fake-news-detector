from transformers import pipeline
import streamlit as st

# Inicializa o modelo
detector = pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect")

label_map = {
    "LABEL_0": "FAKE",
    "LABEL_1": "REAL"
}

# Cria a interface do Streamlit
st.title("Detector de Notícias Falsas")

# Define o limite de caracteres
MAX_CHARS = 800

# Inicializa a variável de sessão para o texto se não existir
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Função para limpar o texto
def clear_text():
    st.session_state.user_input = ""

# Cria área de texto para entrada do usuário com contador de caracteres
user_text = st.text_area(
    "Digite o conteúdo da notícia para análise:", 
    height=300,
    max_chars=MAX_CHARS,
    help=f"Limite de {MAX_CHARS} caracteres para evitar problemas de processamento.",
    key="user_input"
)

# Calcula o número de caracteres (usado apenas para verificação)
char_count = len(user_text)

# Cria duas colunas para os botões lado a lado
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

# Botão Analisar na coluna da esquerda
with col1:
    analisar_button = st.button("Analisar")

# Botão Limpar na coluna da direita
with col7:
    st.button('Limpar', on_click=clear_text, key='limpar_btn')

# Lógica para o botão Analisar
if analisar_button:
    if user_text.strip() == "":
        st.warning("Por favor, digite algum texto para analisar.")
    elif char_count > MAX_CHARS:
        st.error(f"O texto excede o limite de {MAX_CHARS} caracteres. Por favor, reduza o tamanho do texto.")
    else:
        # Mostra um spinner durante o processamento
        with st.spinner("Analisando o texto..."):
            try: 
                # Obtém a previsão
                result = detector(user_text)[0]
                
                # Mapeia o rótulo
                label = label_map.get(result['label'], result['label'])
                
                # Calcula a porcentagem de confiança
                confidence_percentage = result['score'] * 100
                
                # Exibe o resultado com estilo apropriado
                if label == "FAKE":
                    st.error(f"Esta é provavelmente uma notícia FALSA com {confidence_percentage:.2f}% de confiança.")
                else:
                    st.success(f"Esta é provavelmente uma notícia VERDADEIRA com {confidence_percentage:.2f}% de confiança.")
                
                # Mostra detalhes adicionais
                st.info("Observação: Esta é uma previsão de IA e não deve ser o único fator na determinação da autenticidade da notícia.")
            except Exception as e:
                st.error(f"Ocorreu um erro durante a análise: {str(e)}")
                st.info("Dica: Tente reiniciar a aplicação ou usar um texto mais curto.")
# Adiciona o footer com os direitos reservados
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; padding: 10px;'>Todos os direitos reservados | Misael Lima, Efraim Ferreira e Ítalo Santos - 2025</div>", unsafe_allow_html=True)
