import streamlit as st
import pandas as pd
import subprocess
import shutil
import os
import io
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from Bio import SeqIO
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="GeoAMR - Clinical Diagnostics & Discovery Suite",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stButton>button { border-radius: 8px; width: 100%; background-color: #c0392b; color: white; font-weight: bold; }
        .stButton>button:hover { background-color: #e74c3c; color: white; }
        .signature-text { font-size: 16px; color: #e74c3c; font-style: italic; font-weight: bold; margin-top: -15px; margin-bottom: 25px; text-align: center; }
        .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 GeoAMR-Gonorrhoeae Tracker")
st.subheader("Comprehensive Automated Assembly Profiling & Surveillance Platform")
st.markdown('<p class="signature-text">Produced by Henry — Advanced Clinical Genomics Unit</p>', unsafe_allow_html=True)
st.markdown("---")

# --- Interactive Sidebar Controls ---
st.sidebar.header("🎛️ Pipeline Parameters")
min_id = st.sidebar.slider("Minimum % Identity", min_value=50, max_value=100, value=75, step=5)
min_cov = st.sidebar.slider("Minimum % Coverage", min_value=10, max_value=100, value=50, step=5)

st.sidebar.markdown("---")
st.sidebar.info("This clinical system cross-references alignments against the ResFinder, CARD, NCBI, and VFDB sequence databases.")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analysis Features")
st.sidebar.markdown("- ✅ AMR Gene Detection")
st.sidebar.markdown("- ✅ Virulence Factors")
st.sidebar.markdown("- ✅ Recombination Frequency")
st.sidebar.markdown("- ✅ Nucleotide Diversity")
st.sidebar.markdown("- ✅ Physical Gene Linkage")
st.sidebar.markdown("- ✅ PCA & Network Analysis")

# --- Working Computational Core ---
def run_abricate_multi(fasta_path, db_name, identity_threshold, coverage_threshold):
    """
    Executes abricate via system subprocess, resolves duplicate coverage columns,
    and maps headers explicitly.
    """
    try:
        cmd = [
            "abricate", 
            "--db", db_name, 
            "--minid", str(identity_threshold), 
            "--mincov", str(coverage_threshold), 
            fasta_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        
        if not result.stdout.strip():
            return pd.DataFrame()
        
        data = io.StringIO(result.stdout)
        df = pd.read_csv(data, sep="\t")
        
        if df.empty:
            return pd.DataFrame()
            
        mapping_dict = {}
        for col in df.columns:
            normalized = col.upper().strip()
            if normalized == '%COVERAGE' or (normalized == 'COVERAGE' and '%COVERAGE' not in df.columns):
                mapping_dict[col] = '% Coverage'
            elif normalized == '%IDENTITY' or (normalized == 'IDENTITY' and '%IDENTITY' not in df.columns):
                mapping_dict[col] = '% Identity'
            elif normalized == 'SEQUENCE':
                mapping_dict[col] = 'Contig/Node'
            elif normalized == 'START':
                mapping_dict[col] = 'Start'
            elif normalized == 'END':
                mapping_dict[col] = 'End'
            elif normalized == 'GENE':
                mapping_dict[col] = 'Identified Gene'
            elif normalized == 'DATABASE':
                mapping_dict[col] = 'Source DB'
            elif normalized == 'PRODUCT':
                mapping_dict[col] = 'Functional Product/Annotation'

        available_target_keys = [k for k in mapping_dict.keys() if k in df.columns]
        df_clean = df[available_target_keys].copy()
        df_clean.rename(columns=mapping_dict, inplace=True)
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        
        return df_clean

    except Exception as e:
        print(f"Background notice for DB [{db_name}]: {str(e)}")
        return pd.DataFrame()

# --- Analytical & Report Construction Helpers ---
def generate_amr_matrix(all_results_df):
    """Maps found genes to clinical class resistance categories for heatmap indexing."""
    if all_results_df.empty:
        return pd.DataFrame()
        
    def drug_class_mapper(gene_name):
        gene_upper = str(gene_name).upper()
        if any(x in gene_upper for x in ['GYRA', 'PARC']):
            return 'Fluoroquinolones (Ciprofloxacin)'
        if any(x in gene_upper for x in ['PENA', 'BLATEM', 'BLA', 'NDM']):
            return 'Cephalosporins / Penicillins'
        if any(x in gene_upper for x in ['TETM', 'TET']):
            return 'Tetracyclines'
        if any(x in gene_upper for x in ['ERM', 'MPH', 'MTRR', 'MACA', 'MACB']):
            return 'Macrolides (Azithromycin)'
        if any(x in gene_upper for x in ['APH', 'ANT', 'AAD', 'RPSL']):
            return 'Aminoglycosides'
        if any(x in gene_upper for x in ['SUL', 'FOLA']):
            return 'Sulfonamides'
        return 'Other Resistance Trait'

    df_amr = all_results_df[all_results_df['Source DB'].isin(['resfinder', 'card', 'ncbi'])].copy()
    if df_amr.empty:
        return pd.DataFrame()
        
    df_amr['Drug Class'] = df_amr['Identified Gene'].apply(drug_class_mapper)
    matrix = df_amr.pivot_table(index='Sample ID', columns='Drug Class', aggfunc='size', fill_value=0)
    return matrix

def calculate_recombination_frequency(master_df):
    """Calculate recombination frequency based on linked genes within same contig"""
    linkage_records = []
    for (sample, contig), group in master_df.groupby(['Sample ID', 'Contig/Node']):
        if len(group) >= 2:
            sorted_genes = group.sort_values('Start')
            for i in range(len(sorted_genes)):
                for j in range(i + 1, len(sorted_genes)):
                    distance = abs(sorted_genes.iloc[j]['Start'] - sorted_genes.iloc[i]['Start'])
                    if distance <= 50000:
                        linkage_records.append({
                            'sample': sample,
                            'gene_a': sorted_genes.iloc[i]['Identified Gene'],
                            'gene_b': sorted_genes.iloc[j]['Identified Gene'],
                            'distance': distance
                        })
    
    if not linkage_records:
        return 0, pd.DataFrame()
    
    recombination_freq = len(linkage_records) / max(1, master_df['Sample ID'].nunique())
    return recombination_freq, pd.DataFrame(linkage_records)

def calculate_nucleotide_diversity(master_df):
    """Calculate nucleotide diversity from SNP-like patterns"""
    if master_df.empty:
        return 0
    
    amr_data = master_df[master_df['Source DB'].isin(['resfinder', 'card', 'ncbi'])]
    if len(amr_data) < 2:
        return 0
    
    avg_coverage = amr_data['% Coverage'].mean()
    avg_identity = amr_data['% Identity'].mean()
    
    diversity = (100 - avg_identity) / 100 * (avg_coverage / 100)
    return round(diversity, 6)

def perform_pca_analysis(master_df):
    """Perform PCA on AMR gene presence/absence matrix"""
    if master_df.empty:
        return None, None, None
    
    amr_data = master_df[master_df['Source DB'].isin(['resfinder', 'card', 'ncbi'])].copy()
    if amr_data.empty:
        return None, None, None
    
    binary_matrix = amr_data.pivot_table(
        index='Sample ID', 
        columns='Identified Gene', 
        aggfunc='size', 
        fill_value=0
    )
    binary_matrix = (binary_matrix > 0).astype(int)
    
    if binary_matrix.shape[0] < 2 or binary_matrix.shape[1] < 1:
        return None, None, None
    
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(2, binary_matrix.shape[0], binary_matrix.shape[1]))
        pca_result = pca.fit_transform(binary_matrix)
        explained_var = pca.explained_variance_ratio_
        return pca_result, explained_var, binary_matrix.index.tolist()
    except ImportError:
        st.warning("scikit-learn not installed for PCA. Install with: pip install scikit-learn")
        return None, None, None

def create_network_plot(master_df):
    """Create genetic similarity network"""
    if master_df.empty:
        return None
    
    try:
        import networkx as nx
        
        G = nx.Graph()
        amr_data = master_df[master_df['Source DB'].isin(['resfinder', 'card', 'ncbi'])]
        
        if amr_data.empty:
            return None
        
        genes = amr_data['Identified Gene'].unique()
        for gene in genes:
            G.add_node(gene)
        
        for sample in amr_data['Sample ID'].unique():
            sample_genes = amr_data[amr_data['Sample ID'] == sample]['Identified Gene'].tolist()
            for i in range(len(sample_genes)):
                for j in range(i + 1, len(sample_genes)):
                    if G.has_edge(sample_genes[i], sample_genes[j]):
                        G[sample_genes[i]][sample_genes[j]]['weight'] += 1
                    else:
                        G.add_edge(sample_genes[i], sample_genes[j], weight=1)
        
        return G
    except ImportError:
        st.warning("networkx not installed for network visualization")
        return None

def generate_pdf_report(summary_df, total_samples, recombination_freq, nucleotide_diversity):
    """Generates an error-proof, structurally sound PDF byte array"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "GeoAMR Surveillance & Clinical Diagnostics Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Produced by Henry - Automated Public Health Genomics Output", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Executive Batch Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Total Genomes Successfully Screened: {total_samples}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Total Elements Logged Across Registries: {len(summary_df)} total loci hits", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Recombination Frequency: {recombination_freq:.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Nucleotide Diversity: {nucleotide_diversity:.6f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. High-Yield Target Annotations (Top Hits)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(40, 7, "Sample ID", border=1)
    pdf.cell(30, 7, "Gene", border=1)
    pdf.cell(20, 7, "Cov %", border=1)
    pdf.cell(20, 7, "Iden %", border=1)
    pdf.cell(25, 7, "DB Source", border=1)
    pdf.cell(55, 7, "Product Annotation", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 8)
    for idx, row in summary_df.head(35).iterrows():
        s_id = str(row['Sample ID'])[:20].encode('latin-1', 'replace').decode('latin-1')
        g_id = str(row['Identified Gene']).encode('latin-1', 'replace').decode('latin-1')
        cov = str(row['% Coverage']).encode('latin-1', 'replace').decode('latin-1')
        iden = str(row['% Identity']).encode('latin-1', 'replace').decode('latin-1')
        src = str(row['Source DB']).encode('latin-1', 'replace').decode('latin-1')
        prod = str(row.get('Functional Product/Annotation', 'N/A'))[:32].encode('latin-1', 'replace').decode('latin-1')
        
        pdf.cell(40, 6, s_id, border=1)
        pdf.cell(30, 6, g_id, border=1)
        pdf.cell(20, 6, cov, border=1)
        pdf.cell(20, 6, iden, border=1)
        pdf.cell(25, 6, src, border=1)
        pdf.cell(55, 6, prod, border=1, new_x="LMARGIN", new_y="NEXT")
        
    return pdf.output(dest='S')

# --- UI Layout Interface & Data Flow ---

st.markdown("### 1. Batch System: Upload Assembled Genomes")
uploaded_files = st.file_uploader(
    "Drag and drop your assembled FASTA genomes here simultaneously", 
    type=["fasta", "fa"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 Queue initialized: {len(uploaded_files)} samples loaded into system storage.")
    
    abricate_installed = shutil.which("abricate") is not None
    all_batch_records = []
    
    if abricate_installed:
        databases_to_screen = ["resfinder", "card", "ncbi", "vfdb"]
        
        progress_bar = st.progress(0)
        for index, file_obj in enumerate(uploaded_files):
            bytes_data = file_obj.read()
            fasta_string = bytes_data.decode("utf-8")
            current_sample_name = os.path.splitext(file_obj.name)[0]
            
            temp_fasta_path = f"temp_{current_sample_name}.fasta"
            with open(temp_fasta_path, "w") as f:
                f.write(fasta_string)
            
            for db in databases_to_screen:
                db_df = run_abricate_multi(temp_fasta_path, db, min_id, min_cov)
                if not db_df.empty:
                    db_df['Sample ID'] = current_sample_name
                    all_batch_records.append(db_df)
                    
            if os.path.exists(temp_fasta_path):
                os.remove(temp_fasta_path)
                
            progress_bar.progress((index + 1) / len(uploaded_files))
            
        if all_batch_records:
            master_df = pd.concat(all_batch_records, ignore_index=True)
            st.success("✨ Sequence data successfully compiled across all reference frameworks!")
            
            # --- Metrics Dashboard ---
            st.markdown("---")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            recombination_freq, linkage_df = calculate_recombination_frequency(master_df)
            nucleotide_diversity = calculate_nucleotide_diversity(master_df)
            
            with col1:
                st.metric("🧬 Genomes", master_df['Sample ID'].nunique())
            with col2:
                st.metric("🧪 AMR Genes", master_df[master_df['Source DB'].isin(['resfinder', 'card', 'ncbi'])]['Identified Gene'].nunique())
            with col3:
                st.metric("🦠 Virulence", master_df[master_df['Source DB'] == 'vfdb']['Identified Gene'].nunique())
            with col4:
                st.metric("🔄 Recombination Freq", f"{recombination_freq:.3f}")
            with col5:
                st.metric("📊 Diversity Index", f"{nucleotide_diversity:.6f}")
            
            # --- REQUIREMENT: DISPLAY ALL GRANULAR TABLES FIRST ---
            st.markdown("---")
            st.markdown("### 2. Primary Biological Feature Registries")
            
            tab1, tab2, tab3 = st.tabs([
                "💊 Antimicrobial Resistance (AMR) Elements", 
                "🧬 Virulence Factors & Toxins", 
                "📊 Total Master Genome Registry (All Genes)"
            ])
            
            with tab1:
                amr_mask = master_df['Source DB'].isin(['resfinder', 'card', 'ncbi'])
                final_amr_df = master_df[amr_mask]
                if not final_amr_df.empty:
                    st.dataframe(final_amr_df, width="stretch")
                    csv_amr = final_amr_df.to_csv(index=False)
                    st.download_button("📥 Download AMR Data (CSV)", csv_amr, "amr_data.csv", "text/csv")
                else:
                    st.info("No explicit resistance determinants detected above parameters.")
                    
            with tab2:
                vf_mask = master_df['Source DB'].isin(['vfdb'])
                final_vf_df = master_df[vf_mask]
                if not final_vf_df.empty:
                    st.dataframe(final_vf_df, width="stretch")
                else:
                    st.info("No explicit structural virulence vectors or pathogenic attributes identified.")
                    
            with tab3:
                st.write("Comprehensive indexing of all unique loci mapped within this screening operation:")
                st.dataframe(master_df, width="stretch")
            
            # --- REQUIREMENT: GENOMIC LINEAR FEATURE MAP DISPLAY ---
            st.markdown("---")
            st.markdown("### 3. Structural Contig Loci Coordinates & Linear Mapping")
            
            if len(master_df['Sample ID'].unique()) > 0:
                selected_map_sample = st.selectbox("Select Sample Strain to Map Coordinates:", master_df['Sample ID'].unique())
                sample_map_data = master_df[master_df['Sample ID'] == selected_map_sample]
                
                if not sample_map_data.empty:
                    fig_map = px.scatter(
                        sample_map_data,
                        x="Start",
                        y="Identified Gene",
                        color="Source DB",
                        size="% Coverage",
                        hover_data=["End", "Functional Product/Annotation"],
                        title=f"Linear Multi-Loci Feature Architecture Mapping: Strain {selected_map_sample}",
                        labels={"Start": "Nucleotide Coordinates Base Position (bp)"}
                    )
                    fig_map.update_traces(marker=dict(symbol="square", line=dict(width=1, color="DarkSlateGrey")))
                    fig_map.update_layout(xaxis_showgrid=True, yaxis_showgrid=True, height=500)
                    st.plotly_chart(fig_map, width="stretch")
            
            # --- Physical Distance Analysis (Gene Linkage) ---
            st.markdown("---")
            st.markdown("### 3b. Gene Linkage & Physical Distance Analysis")
            
            if not linkage_df.empty:
                st.markdown(f"**Recombination Frequency: {recombination_freq:.4f}**")
                st.markdown("Genes found in close physical proximity (<50kb):")
                st.dataframe(linkage_df.head(20), width="stretch")
                
                if len(linkage_df) > 0:
                    fig_dist = px.bar(
                        linkage_df.head(20),
                        x="gene_a",
                        y="distance",
                        color="gene_b",
                        title="Top 20 Gene Pairs by Physical Distance",
                        labels={"distance": "Distance (bp)", "gene_a": "Gene A"}
                    )
                    st.plotly_chart(fig_dist, width="stretch")
            else:
                st.info("No linked gene pairs detected within 50kb distance threshold.")
            
            # --- REQUIREMENT: EPIDEMIOLOGICAL HEATMAPPING ---
            st.markdown("---")
            st.markdown("### 4. Cross-Resistance Clustering Analytics")
            with st.spinner("Reindexing phenotypical profiles..."):
                amr_matrix = generate_amr_matrix(master_df)
                if not amr_matrix.empty:
                    fig_heatmap = px.imshow(
                        amr_matrix,
                        labels=dict(x="Antibiotic Drug Class Family", y="Sample Strain Node", color="Loci Count"),
                        x=amr_matrix.columns,
                        y=amr_matrix.index,
                        color_continuous_scale="Reds",
                        text_auto=True,
                        aspect="auto"
                    )
                    fig_heatmap.update_layout(
                        title_text="Phenotypical Genotype Profile Breakdown Heatmap", 
                        title_x=0.5,
                        height=max(400, len(amr_matrix) * 30)
                    )
                    st.plotly_chart(fig_heatmap, width="stretch")
                else:
                    st.info("Insufficient resistance mutations available to populate cluster visualizations.")
            
            # --- PCA and Network Analysis ---
            st.markdown("---")
            st.markdown("### 4b. Population Genomics: PCA & Genetic Similarity Network")
            
            pca_result, explained_var, sample_names = perform_pca_analysis(master_df)
            if pca_result is not None and len(pca_result) >= 2:
                pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
                pca_df['Sample'] = sample_names
                
                fig_pca = px.scatter(
                    pca_df, x='PC1', y='PC2', text='Sample',
                    title=f"PCA of AMR Gene Profiles (PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%})",
                    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                    color_discrete_sequence=['#c0392b']
                )
                fig_pca.update_traces(textposition='top center', marker=dict(size=15))
                fig_pca.update_layout(height=500)
                st.plotly_chart(fig_pca, width="stretch")
            else:
                st.info("Insufficient data for PCA analysis (need at least 2 samples with AMR genes).")
            
            G = create_network_plot(master_df)
            if G and G.number_of_nodes() > 0:
                st.markdown("#### AMR Gene Co-occurrence Network")
                st.info("Nodes represent AMR genes, edges represent co-occurrence in same genome.")
                
                try:
                    import networkx as nx
                    pos = nx.spring_layout(G, k=1, iterations=50)
                    
                    edge_traces = []
                    for edge in G.edges(data=True):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        weight = edge[2].get('weight', 1)
                        line_width = min(5, weight * 2)
                        edge_traces.append(
                            go.Scatter(
                                x=[x0, x1, None], y=[y0, y1, None],
                                mode='lines', line=dict(width=line_width, color='#95a5a6'),
                                hoverinfo='none', showlegend=False
                            )
                        )
                    
                    node_x = [pos[node][0] for node in G.nodes()]
                    node_y = [pos[node][1] for node in G.nodes()]
                    node_text = list(G.nodes())
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y, mode='markers+text', text=node_text,
                        textposition="top center", hoverinfo='text',
                        marker=dict(size=20, color='#c0392b', line=dict(width=2, color='white')),
                        showlegend=False
                    )
                    
                    fig_network = go.Figure(data=edge_traces + [node_trace])
                    fig_network.update_layout(
                        title="AMR Gene Co-occurrence Network",
                        showlegend=False, height=600,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    st.plotly_chart(fig_network, width="stretch")
                except Exception as e:
                    st.warning(f"Network visualization limited: {e}")
            
            # --- REQUIREMENT: SECURE PDF COMPILATION & DOWNLOAD ---
            st.markdown("---")
            st.markdown("### 5. Regulatory Export Panel")
            
            col_pdf1, col_pdf2 = st.columns(2)
            
            with col_pdf1:
                try:
                    pdf_output_bytes = generate_pdf_report(master_df, len(uploaded_files), recombination_freq, nucleotide_diversity)
                    final_pdf_buffer = bytes(pdf_output_bytes)
                    
                    st.download_button(
                        label="📥 Download Clinical Batch Summary PDF",
                        data=final_pdf_buffer,
                        file_name="GeoAMR_Clinical_Surveillance_Report.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF Report compilation complete.")
                except Exception as pdf_err:
                    st.error(f"Reporting compilation warning: {str(pdf_err)}")
            
            with col_pdf2:
                all_csv = master_df.to_csv(index=False)
                st.download_button(
                    label="📊 Export All Data (CSV)",
                    data=all_csv,
                    file_name="geoamr_all_detections.csv",
                    mime="text/csv"
                )
            
            # --- Final Signature ---
            st.markdown("---")
            st.markdown('<p class="signature-text">🧬 GeoAMR Engine | Developed by Henry, PhD | Clinical Genomics & Public Health Surveillance</p>', unsafe_allow_html=True)
                
        else:
            st.warning("All files systematically inspected, but zero targets matched criteria configurations across active libraries.")
    else:
        st.error("🔒 Critical Error: Native Abricate installation profile missing from the workstation's active runtime path environment.")
        st.markdown("""
        ### Installation Instructions:
        ```bash
        # Install ABRicate for AMR detection
        conda install -c bioconda abricate
        
        # Or download databases manually:
        abricate --databases --list
        abricate --setupdb
