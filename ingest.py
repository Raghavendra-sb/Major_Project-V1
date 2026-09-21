"""
ingest.py
---------
Creates sample skin disease knowledge documents and ingests them
into a Qdrant vector store using HuggingFace embeddings.

Run this ONCE before starting app.py:
    python ingest.py

Later, replace SKIN_DISEASE_KNOWLEDGE with real PDFs using:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader("your_skin_disease.pdf")
    docs = loader.load()
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Sample knowledge base (replace with real PDFs when available)
# ─────────────────────────────────────────────────────────────
SKIN_DISEASE_KNOWLEDGE = [

    Document(
        page_content="""
        Acne (Acne Vulgaris)
        Acne is one of the most common skin conditions affecting millions worldwide.
        It occurs when hair follicles become plugged with oil and dead skin cells.

        Causes:
        - Excess sebum (oil) production by sebaceous glands
        - Proliferation of Cutibacterium acnes bacteria
        - Hormonal changes (androgens increase oil production)
        - Dead skin cell accumulation blocking pores
        - Diet high in refined sugars may worsen acne
        - Stress can trigger hormonal fluctuations

        Symptoms:
        - Whiteheads, blackheads, papules, pustules, nodules, cysts
        - Inflammation and redness around lesions
        - Scarring in severe or untreated cases

        Treatments:
        - Topical retinoids (tretinoin, adapalene) to unclog pores
        - Benzoyl peroxide to kill bacteria and reduce oil
        - Salicylic acid for gentle exfoliation
        - Oral antibiotics (doxycycline, minocycline) for moderate to severe acne
        - Isotretinoin (Accutane) for severe cystic acne
        - Hormonal therapy (oral contraceptives) for females

        Prevention:
        - Wash face twice daily with a gentle, non-comedogenic cleanser
        - Avoid touching or popping pimples to prevent scarring
        - Use oil-free, non-comedogenic moisturizers and cosmetics
        - Change pillowcases regularly
        - Maintain a balanced diet and stay hydrated
        """,
        metadata={"disease": "Acne", "source": "skin_knowledge_base", "page_label": "1"}
    ),

    Document(
        page_content="""
        Actinic Keratosis
        Actinic keratosis (AK) is a rough, scaly patch of skin caused by years of sun exposure.
        It is considered a precancerous condition that can develop into squamous cell carcinoma.

        Causes:
        - Cumulative ultraviolet (UV) radiation from sunlight
        - Use of tanning beds
        - Fair skin, light hair, and light eyes increase risk
        - Weakened immune system
        - Age (more common in people over 40)

        Symptoms:
        - Rough, dry, scaly patch (typically less than 1 inch in diameter)
        - Flat to slightly raised patch with a hard, wart-like surface
        - Color varies: pink, red, or brown
        - Itching, burning, or tenderness in the affected area

        Treatments:
        - Cryotherapy: freezing with liquid nitrogen
        - Topical medications: fluorouracil (5-FU), imiquimod, diclofenac
        - Photodynamic therapy (PDT)
        - Chemical peels and laser therapy
        - Surgical removal for isolated lesions

        Prevention:
        - Use broad-spectrum SPF 30+ sunscreen daily
        - Wear protective clothing: hats, long sleeves
        - Avoid peak sun hours (10am to 4pm)
        - Never use tanning beds
        - Annual skin check-ups with a dermatologist
        """,
        metadata={"disease": "Actinic_Keratosis", "source": "skin_knowledge_base", "page_label": "2"}
    ),

    Document(
        page_content="""
        Benign Skin Tumors
        Benign tumors of the skin are non-cancerous growths that do not invade nearby tissue
        or spread to other parts of the body.

        Common types:
        - Seborrheic keratosis: waxy, wart-like growths
        - Dermatofibromas: small, firm bumps often on the legs
        - Lipomas: soft, fatty lumps under the skin
        - Hemangiomas: blood vessel growths (often in infants)
        - Nevi (moles): pigmented skin growths

        Causes:
        - Genetic predisposition
        - Sun exposure (for some types)
        - Hormonal changes
        - Minor skin injuries

        Symptoms:
        - A growth or lump on or under the skin
        - May be soft, firm, smooth, or rough
        - Usually painless, no ulceration or bleeding

        Treatments:
        - Most benign tumors do not require treatment
        - Surgical excision if cosmetically bothersome or irritated
        - Laser removal or cryotherapy for surface lesions
        - Regular monitoring for any changes in size, color, or shape

        When to see a doctor immediately:
        - Rapid growth in size
        - Change in color or shape
        - Bleeding or ulceration
        - Pain or tenderness
        """,
        metadata={"disease": "Benign_tumors", "source": "skin_knowledge_base", "page_label": "3"}
    ),

    Document(
        page_content="""
        Eczema (Atopic Dermatitis)
        Eczema is a chronic inflammatory skin condition characterized by itchy,
        red, and cracked skin. It commonly starts in childhood but can affect any age.

        Causes:
        - Genetic mutations affecting the skin barrier protein filaggrin
        - Overactive immune system response to environmental triggers
        - Environmental allergens: dust mites, pet dander, pollen
        - Irritants: soaps, detergents, perfumes, certain fabrics
        - Stress and hormonal changes
        - Food allergies (eggs, milk, peanuts in some cases)

        Symptoms:
        - Dry, sensitive skin and intense itching especially at night
        - Red to brownish-gray patches
        - Small, raised bumps that may weep fluid when scratched
        - Thickened, cracked, or scaly skin

        Treatments:
        - Moisturizers applied immediately after bathing (emollients)
        - Topical corticosteroids to reduce inflammation
        - Topical calcineurin inhibitors (tacrolimus, pimecrolimus)
        - Dupilumab (biologic injection) for moderate-to-severe eczema
        - Antihistamines to relieve itching

        Prevention and management:
        - Moisturize skin at least twice daily
        - Identify and avoid personal triggers
        - Use mild, fragrance-free soaps and detergents
        - Keep fingernails short to minimize scratching damage
        - Wear soft, breathable fabrics like cotton
        """,
        metadata={"disease": "Eczema", "source": "skin_knowledge_base", "page_label": "4"}
    ),

    Document(
        page_content="""
        Lupus (Cutaneous Lupus Erythematosus)
        Lupus is a chronic autoimmune disease where the immune system attacks healthy tissue,
        including the skin. Cutaneous lupus specifically affects the skin.

        Types:
        - Discoid lupus erythematosus (DLE): chronic, scarring skin lesions
        - Subacute cutaneous lupus: non-scarring lesions, photosensitive
        - Acute cutaneous lupus: butterfly (malar) rash across cheeks and nose

        Causes:
        - Autoimmune dysfunction (immune system attacks own tissues)
        - Genetic susceptibility (HLA gene variants)
        - UV light exposure triggers or worsens symptoms
        - Hormonal factors (more common in women of childbearing age)
        - Certain medications can trigger drug-induced lupus

        Symptoms:
        - Butterfly-shaped malar rash on face
        - Photosensitivity (skin reacts badly to sunlight)
        - Discoid coin-shaped lesions that can scar
        - Hair loss near lesions and oral ulcers

        Treatments:
        - Topical corticosteroids and calcineurin inhibitors
        - Antimalarial drugs: hydroxychloroquine (Plaquenil)
        - Systemic immunosuppressants for severe cases
        - Strict sun protection: SPF 50+, protective clothing
        - Avoid known triggers

        Monitoring:
        - Regular blood tests to monitor organ involvement
        - Annual ophthalmology check for hydroxychloroquine users
        - Dermatology and rheumatology follow-ups every 3-6 months
        """,
        metadata={"disease": "Lupus", "source": "skin_knowledge_base", "page_label": "5"}
    ),

    Document(
        page_content="""
        Skin Cancer
        Skin cancer is the abnormal growth of skin cells, most often developing on skin
        exposed to sunlight. There are three main types: basal cell carcinoma (BCC),
        squamous cell carcinoma (SCC), and melanoma.

        Causes:
        - UV radiation from sun or tanning beds (primary cause)
        - Fair skin, multiple moles, family history of skin cancer
        - Weakened immune system
        - Exposure to radiation or certain chemicals like arsenic
        - History of sunburns especially in childhood

        Symptoms - ABCDE rule for melanoma:
        - Asymmetry: one half does not match the other
        - Border: irregular, ragged, notched edges
        - Color: multiple shades of brown, black, red, white, or blue
        - Diameter: larger than 6mm (size of a pencil eraser)
        - Evolving: changing in size, shape, or color

        Treatments:
        - Surgical excision (most common treatment)
        - Mohs surgery for BCC and SCC (skin-sparing technique)
        - Radiation therapy and photodynamic therapy
        - Immunotherapy (pembrolizumab, nivolumab) for advanced melanoma
        - Targeted therapy (BRAF/MEK inhibitors) for melanoma with BRAF mutation
        - Chemotherapy for advanced cases

        Prevention:
        - Use SPF 30+ broad-spectrum sunscreen daily, reapply every 2 hours
        - Perform monthly self-skin examinations
        - Annual professional skin checks after age 40
        - Never use tanning beds
        - Seek shade during peak UV hours (10am to 4pm)
        """,
        metadata={"disease": "SkinCancer", "source": "skin_knowledge_base", "page_label": "6"}
    ),

    Document(
        page_content="""
        Vasculitis (Cutaneous Vasculitis)
        Vasculitis is inflammation of blood vessels. Cutaneous vasculitis specifically
        affects the blood vessels of the skin causing characteristic skin findings.

        Causes:
        - Autoimmune disorders (immune complexes depositing in vessel walls)
        - Infections: hepatitis B, hepatitis C, streptococcal infections
        - Medications: antibiotics, NSAIDs, diuretics
        - Inflammatory diseases: lupus, rheumatoid arthritis
        - Idiopathic (unknown cause) in many cases

        Symptoms:
        - Palpable purpura: raised purple-red spots (most characteristic sign)
        - Urticaria (hives) that last more than 24 hours
        - Livedo reticularis: mottled net-like skin discoloration
        - Skin ulcers in severe cases
        - Systemic symptoms: joint pain, fever, fatigue

        Treatments:
        - Identify and remove the trigger (stop offending medication, treat infection)
        - NSAIDs for mild symptoms
        - Corticosteroids (prednisone) for moderate to severe disease
        - Immunosuppressants (azathioprine, methotrexate) for chronic cases
        - Colchicine for urticarial vasculitis
        - Biologic agents for refractory cases

        Monitoring:
        - Regular urinalysis and kidney function tests
        - Watch for systemic involvement of kidneys, lungs, or nerves
        - Follow-up with rheumatologist and dermatologist every 3-6 months
        """,
        metadata={"disease": "Vasculitis", "source": "skin_knowledge_base", "page_label": "7"}
    ),

    Document(
        page_content="""
        Warts (Verruca)
        Warts are non-cancerous skin growths caused by human papillomavirus (HPV) infection.
        They are contagious and can spread through direct or indirect contact.

        Types:
        - Common warts (verruca vulgaris): rough, grainy growths usually on hands
        - Plantar warts: hard, grainy growths on heels or balls of feet
        - Flat warts: flat-topped, slightly raised lesions on face or legs
        - Filiform warts: thread-like growths on face, neck, or eyelids

        Causes:
        - Human papillomavirus (HPV) infection through cuts or breaks in skin
        - Direct contact with a wart or contaminated surface
        - Walking barefoot in public places (for plantar warts)
        - Weakened immune system increases susceptibility

        Symptoms:
        - Flesh-colored, rough bumps on the skin
        - Black pinpoints (clotted blood vessels) visible inside wart
        - Tender when pressed (especially plantar warts)
        - Can grow in clusters (mosaic warts)

        Treatments:
        - Salicylic acid: over-the-counter peeling medicine (most common)
        - Cryotherapy: freezing with liquid nitrogen by a doctor
        - Electrosurgery: burning the wart with electric current
        - Laser treatment for stubborn warts
        - Immunotherapy: stimulating immune system to fight HPV
        - Topical imiquimod cream

        Prevention:
        - Avoid touching warts (yours or others)
        - Keep hands clean and dry
        - Wear footwear in public showers and pools
        - Do not share personal items like towels or razors
        - Get HPV vaccination to protect against some strains
        """,
        metadata={"disease": "Warts", "source": "skin_knowledge_base", "page_label": "8"}
    ),
]


def run_ingestion():
    print("🚀 Starting ingestion into Qdrant...")

    # ── Chunk documents ──
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = splitter.split_documents(SKIN_DISEASE_KNOWLEDGE)
    print(f"📄 Total chunks created: {len(split_docs)}")

    # ── Embeddings (runs locally, no API key needed) ──
    print("🔄 Loading HuggingFace embedding model (first run downloads ~80MB)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Push to Qdrant ──
    print("📦 Pushing chunks to Qdrant vector store...")
    QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name="skin_disease_vectors",
        embedding=embedding_model,
    )

    print("✅ Ingestion complete! Collection name: skin_disease_vectors")


if __name__ == "__main__":
    run_ingestion()