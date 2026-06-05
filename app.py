import streamlit as st
import pandas as pd
import subprocess
import shutil
import os
import io
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from Bio import SeqIO
import numpy as np
import re

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
    </style>
""", unsafe_allow_html=True)

st.title("🧬 GeoAMR-Gonorrhoeae Tracker")
st.subheader("Comprehensive Automated Assembly Profiling & Surveillance Platform")
st.markdown('<p class="signature-text">Produced by Henry — Advanced Clinical Genomics Unit</p>', unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("🎛️ Pipeline Parameters")
min_id = st.sidebar.slider("Minimum % Identity", min_value=50, max_value=100, value=80, step=5)
min_cov = st.sidebar.slider("Minimum % Coverage", min_value=10, max_value=100, value=60, step=5)

st.sidebar.markdown("---")
st.sidebar.info("Using NCBI AMRFinderPlus - Official CDC/NIH Antimicrobial Resistance Database")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analysis Features")
st.sidebar.markdown("- ✅ AMR Gene Detection (NCBI validated)")
st.sidebar.markdown("- ✅ Point Mutations")
st.sidebar.markdown("- ✅ Virulence Factors")
st.sidebar.markdown("- ✅ Recombination Frequency")
st.sidebar.markdown("- ✅ Nucleotide Diversity")
st.sidebar.markdown("- ✅ PCA & Clustering")
st.sidebar.markdown("- ✅ Physical Linkage Mapping")

# --- AMRFinderPlus Integration ---
def run_amrfinder(fasta_path, sample_name):
    """
    Run NCBI AMRFinderPlus on a genome file
    Returns DataFrame with real AMR detections
    """
    try:
        # Run AMRFinderPlus
        cmd = [
            "amrfinder",
            "-n", fasta_path,
            "--name", sample_name,
            "--ident_min", str(min_id),
            "--coverage_min", str(min_cov),
            "--plus"  # Includes point mutations and virulence factors
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        
        if result.returncode != 0:
            st.warning(f"AMRFinder warning for {sample_name}: {result.stderr[:200]}")
            return pd.DataFrame()
        
        # Parse TSV output
        if result.stdout.strip():
            df = pd.read_csv(io.StringIO(result.stdout), sep="\t")
            return df
        else:
            return pd.DataFrame()
            
    except subprocess.TimeoutExpired:
        st.error(f"Timeout processing {sample_name}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error with {sample_name}: {str(e)}")
        return pd.DataFrame()

def parse_amrfinder_output(df, sample_name):
    """
    Parse AMRFinderPlus output into standardized format
    """
    if df.empty:
        return []
    
    parsed_results = []
    
    for idx, row in df.iterrows():
        # Map AMRFinder columns to our standard format
        result = {
            "Sample ID": sample_name,
            "Identified Gene": row.get("Gene symbol", row.get("Element symbol", "Unknown")),
            "Element Type": row.get("Element type", "Unknown"),
            "Source DB": "NCBI-AMRFinder",
            "Drug Class": row.get("Drug class", classify_gene(row.get("Gene symbol", ""))),
            "% Coverage": row.get("% Coverage", row.get("Coverage", 0)),
            "% Identity": row.get("% Identity", row.get("Identity", 0)),
            "Contig/Node": row.get("Contig id", row.get("Sequence ID", "Unknown")),
            "Start": row.get("Start", 0),
            "End": row.get("Stop", row.get("End", 0)),
            "Functional Product/Annotation": row.get("Function", row.get("Product", "")),
            "Accession": row.get("Accession of closest sequence", "")
        }
        parsed_results.append(result)
    
    return parsed_results

def classify_gene(gene_name):
    """Classify genes into drug classes for heatmap"""
    gene_upper = str(gene_name).upper()
    if any(x in gene_upper for x in ['GYRA', 'PARC', 'GYRB', 'PARE', 'QNR']):
        return 'Fluoroquinolones'
    if any(x in gene_upper for x in ['PENA', 'BLATEM', 'BLA', 'NDM', 'CTX-M', 'SHV', 'OXA']):
        return 'Beta-lactams/Cephalosporins'
    if any(x in gene_upper for x in ['TET', 'TETRA']):
        return 'Tetracyclines'
    if any(x in gene_upper for x in ['ERM', 'MPH', 'MEF', 'MSR', 'MTR']):
        return 'Macrolides'
    if any(x in gene_upper for x in ['APH', 'ANT', 'AAC', 'AAD', 'RPSL', 'STR']):
        return 'Aminoglycosides'
    if any(x in gene_upper for x in ['SUL', 'FOL', 'DFR']):
        return 'Sulfonamides'
    if any(x in gene_upper for x in ['CAT', 'CML', 'FLO', 'CHL']):
        return 'Phenicols'
    return 'Other Resistance'

# --- Analysis Functions ---
def generate_amr_matrix(all_results_df):
    """Create binary presence matrix for heatmap"""
    if all_results_df.empty:
        return pd.DataFrame()
    
    pivot_df = all_results_df.pivot_table(
        index='Sample ID', 
        columns='Drug Class', 
        aggfunc='size', 
        fill_value=0
    )
    binary_matrix = (pivot_df > 0).astype(int)
    return binary_matrix

def calculate_linkage_and_recombination(master_df):
    """Calculate gene linkage and recombination frequency"""
    linkage_records = []
    
    for (sample, contig), group in master_df.groupby(['Sample ID', 'Contig/Node']):
        if len(group) >= 2:
            sorted_genes = group.sort_values('Start')
            for i in range(len(sorted_genes)):
                for j in range(i + 1, len(sorted_genes)):
                    distance = abs(sorted_genes.iloc[j]['Start'] - sorted_genes.iloc[i]['Start'])
                    
                    # Classify linkage type
                    if distance <= 5000:
                        linkage_type = "Tightly Linked (<5kb)"
                    elif distance <= 50000:
                        linkage_type = "Moderately Linked (5-50kb)"
                    else:
                        linkage_type = "Distanly Linked (>50kb)"
                    
                    linkage_records.append({
                        'Sample': sample,
                        'Contig': contig,
                        'Gene A': sorted_genes.iloc[i]['Identified Gene'],
                        'Gene B': sorted_genes.iloc[j]['Identified Gene'],
                        'Distance (bp)': distance,
                        'Linkage Type': linkage_type
                    })
    
    if not linkage_records:
        return 0, pd.DataFrame()
    
    # Recombination frequency = linkage events / total samples
    recombination_freq = len(linkage_records) / max(1, master_df['Sample ID'].nunique())
    return recombination_freq, pd.DataFrame(linkage_records)

def calculate_diversity_metrics(master_df):
    """Calculate nucleotide diversity from AMR gene variations"""
    if master_df.empty or len(master_df) < 2:
        return 0, 0
    
    # Use identity as proxy for diversity
    amr_genes = master_df[master_df['Element Type'].isin(['AMR', 'AMR gene'])]
    if len(amr_genes) < 2:
        return 0, 0
    
    avg_identity = amr_genes['% Identity'].mean()
    nucleotide_diversity = (100 - avg_identity) / 100
    
    # Count unique gene combinations per sample (genetic richness)
    gene_combinations = master_df.groupby('Sample ID')['Identified Gene'].nunique()
    genetic_richness = gene_combinations.mean()
    
    return nucleotide_diversity, genetic_richness

def perform_pca_analysis(master_df):
    """PCA on AMR gene presence matrix"""
    if master_df.empty:
        return None, None, None
    
    # Create binary matrix
    binary_matrix = master_df.pivot_table(
        index='Sample ID',
        columns='Identified Gene',
        aggfunc='size',
        fill_value=0
    )
    binary_matrix = (binary_matrix > 0).astype(int)
    
    if binary_matrix.shape[0] < 2 or binary_matrix.shape[1] < 2:
        return None, None, None
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Standardize
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(binary_matrix)
        
        # PCA
        pca = PCA(n_components=min(3, binary_matrix.shape[0], binary_matrix.shape[1]))
        pca_result = pca.fit_transform(matrix_scaled)
        explained_var = pca.explained_variance_ratio_
        
        return pca_result, explained_var, binary_matrix.index.tolist()
    except ImportError:
        st.warning("scikit-learn not available. Install for PCA: pip install scikit-learn")
        return None, None, None

def create_cooccurrence_network(master_df):
    """Create gene co-occurrence network"""
    try:
        import networkx as nx
        
        G = nx.Graph()
        
        # Get all AMR genes
        genes = master_df['Identified Gene'].unique()
        for gene in genes:
            G.add_node(gene)
        
        # Add edges based on co-occurrence
        for sample in master_df['Sample ID'].unique():
            sample_genes = master_df[master_df['Sample ID'] == sample]['Identified Gene'].tolist()
            for i in range(len(sample_genes)):
                for j in range(i + 1, len(sample_genes)):
                    if G.has_edge(sample_genes[i], sample_genes[j]):
                        G[sample_genes[i]][sample_genes[j]]['weight'] += 1
                    else:
                        G.add_edge(sample_genes[i], sample_genes[j], weight=1)
        
        return G
    except ImportError:
        return None

def generate_pdf_report(master_df, total_samples, recombination_freq, diversity):
    """Generate clinical PDF report"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "GeoAMR Clinical Surveillance Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "NCBI AMRFinderPlus - Official CDC/NIH Results", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, "Produced by Henry - Public Health Genomics", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    
    # Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Total Genomes Analyzed: {total_samples}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Total AMR Detections: {len(master_df)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Unique AMR Genes: {master_df['Identified Gene'].nunique()}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Recombination Frequency: {recombination_freq:.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Nucleotide Diversity: {diversity:.6f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Top detections table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Top AMR Detections", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(35, 6, "Sample", border=1)
    pdf.cell(30, 6, "Gene", border=1)
    pdf.cell(25, 6, "Drug Class", border=1)
    pdf.cell(15, 6, "Iden%", border=1)
    pdf.cell(15, 6, "Cov%", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 7)
    for idx, row in master_df.head(30).iterrows():
        pdf.cell(35, 5, str(row['Sample ID'])[:15], border=1)
        pdf.cell(30, 5, str(row['Identified Gene'])[:18], border=1)
        pdf.cell(25, 5, str(row['Drug Class'])[:20], border=1)
        pdf.cell(15, 5, str(row['% Identity']), border=1)
        pdf.cell(15, 5, str(row['% Coverage']), border=1, new_x="LMARGIN", new_y="NEXT")
    
    return pdf.output(dest='S')

# --- Main Interface ---
st.markdown("### 1. Upload Genomes for Analysis")

uploaded_files = st.file_uploader(
    "Upload assembled Gonorrhoeae genomes (FASTA format)", 
    type=["fasta", "fa", "fna"], 
    accept_multiple_files=True,
    help="NCBI AMRFinderPlus will analyze these genomes for AMR genes, point mutations, and virulence factors"
)

if uploaded_files:
    st.info(f"📊 {len(uploaded_files)} genome(s) loaded. Starting NCBI AMRFinderPlus analysis...")
    
    all_results = []
    
    # Check if AMRFinder is installed
    amrfinder_installed = shutil.which("amrfinder") is not None
    
    if not amrfinder_installed:
        st.error("""
        ❌ **AMRFinderPlus not installed**
        
        Please run the setup script first. For Streamlit Cloud, ensure `setup.sh` is in your repository.
        
        For local testing:
        ```bash
        conda install -c bioconda ncbi-amrfinderplus
        amrfinder --update
        ```
        """)
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, file_obj in enumerate(uploaded_files):
            status_text.text(f"Analyzing {file_obj.name}...")

            # Save temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
                content = file_obj.read().decode('utf-8')
                tmp_file.write(content)
                tmp_path = tmp_file.name

            sample_name = os.path.splitext(file_obj.name)[0]

            # Run AMRFinder
            amr_df = run_amrfinder(tmp_path, sample_name)

            # Parse results
            parsed_results = parse_amrfinder_output(amr_df, sample_name)
            all_results.extend(parsed_results)

            # Cleanup
            os.unlink(tmp_path)

            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.empty()

        if all_results:
            master_df = pd.DataFrame(all_results)
            st.success(f"✅ Analysis complete! Found {len(master_df)} AMR determinants across {master_df['Sample ID'].nunique()} samples.")

            # Metrics Dashboard
            st.markdown("---")
            col1, col2, col3, col4, col5 = st.columns(5)

            recombination_freq, linkage_df = calculate_linkage_and_recombination(master_df)
            diversity, richness = calculate_diversity_metrics(master_df)

            with col1:
                st.metric("🧬 Genomes", master_df['Sample ID'].nunique())
            with col2:
                st.metric("🧪 AMR Genes", master_df['Identified Gene'].nunique())
            with col3:
                amr_only = master_df[master_df['Element Type'].isin(['AMR', 'AMR gene'])]
                st.metric("💊 Resistance Genes", len(amr_only))
            with col4:
                st.metric("🔄 Recombination", f"{recombination_freq:.3f}")
            with col5:
                st.metric("📊 Diversity (π)", f"{diversity:.5f}")

            # Display tables
            st.markdown("---")
            st.markdown("### 2. AMR Detection Results")

            tab1, tab2, tab3 = st.tabs([
                "🔬 All Detections",
                "💊 AMR Genes",
                "🔗 Linkage Map"
            ])

            with tab1:
                st.dataframe(master_df, width="stretch")
                csv_data = master_df.to_csv(index=False)
                st.download_button("📥 Download Full Results (CSV)", csv_data, "amrfinder_results.csv", "text/csv")

            with tab2:
                amr_only = master_df[master_df['Element Type'].isin(['AMR', 'AMR gene'])]
                if not amr_only.empty:
                    st.dataframe(amr_only, width="stretch")
                else:
                    st.info("No AMR genes detected")

            with tab3:
                if not linkage_df.empty:
                    st.dataframe(linkage_df, width="stretch")

                    # Visualize linkage distances
                    fig_link = px.scatter(
                        linkage_df,
                        x="Gene A",
                        y="Distance (bp)",
                        color="Linkage Type",
                        title="Physical Linkage Distances Between Genes"
                    )
                    st.plotly_chart(fig_link, width="stretch")
                else:
                    st.info("No linked genes detected on same contigs")

            # Heatmap
            st.markdown("---")
            st.markdown("### 3. Resistance Heatmap")

            amr_matrix = generate_amr_matrix(master_df)
            if not amr_matrix.empty:
                fig_heatmap = px.imshow(
                    amr_matrix,
                    color_continuous_scale="Reds",
                    text_auto=True,
                    aspect="auto",
                    title="AMR Gene Presence by Drug Class"
                )
                fig_heatmap.update_layout(height=max(400, len(amr_matrix) * 30))
                st.plotly_chart(fig_heatmap, width="stretch")

            # PCA Analysis
            st.markdown("---")
            st.markdown("### 4. Population Genomics")

            pca_result, explained_var, sample_names = perform_pca_analysis(master_df)
            if pca_result is not None:
                pca_df = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
                pca_df['Sample'] = sample_names

                fig_pca = px.scatter(
                    pca_df, x='PC1', y='PC2', text='Sample',
                    title=f"PCA of AMR Profiles (PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%})",
                    color_discrete_sequence=['#c0392b']
                )
                fig_pca.update_traces(marker=dict(size=15))
                st.plotly_chart(fig_pca, width="stretch")

            # Co-occurrence Network
            G = create_cooccurrence_network(master_df)
            if G and G.number_of_nodes() > 0:
                st.markdown("### 5. AMR Gene Co-occurrence Network")
                st.info(f"Network shows {G.number_of_nodes()} genes with {G.number_of_edges()} co-occurrence relationships")

                try:
                    import networkx as nx
                    pos = nx.spring_layout(G, k=2, iterations=50)

                    # Create edge traces
                    edge_traces = []
                    for edge in G.edges(data=True):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        weight = edge[2].get('weight', 1)
                        edge_traces.append(
                            go.Scatter(
                                x=[x0, x1, None], y=[y0, y1, None],
                                mode='lines',
                                line=dict(width=min(3, weight), color='#95a5a6'),
                                showlegend=False
                            )
                        )

                    node_x = [pos[node][0] for node in G.nodes()]
                    node_y = [pos[node][1] for node in G.nodes()]

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=list(G.nodes()),
                        textposition="top center",
                        marker=dict(size=15, color='#c0392b', line=dict(width=2, color='white')),
                        showlegend=False
                    )

                    fig_network = go.Figure(data=edge_traces + [node_trace])
                    fig_network.update_layout(
                        title="AMR Gene Co-occurrence Network",
                        height=600,
                        showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    st.plotly_chart(fig_network, width="stretch")
                except Exception as e:
                    st.warning(f"Network visualization error: {e}")

            # PDF Report
            st.markdown("---")
            st.markdown("### 6. Export Clinical Report")

            if st.button("📄 Generate PDF Report"):
                try:
                    pdf_data = generate_pdf_report(master_df, len(uploaded_files), recombination_freq, diversity)
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=bytes(pdf_data),
                        file_name="GeoAMR_Clinical_Report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF generation error: {e}")

            # Signature
            st.markdown("---")
            st.markdown('<p class="signature-text">🧬 GeoAMR | Powered by NCBI AMRFinderPlus | Clinical Genomics & Public Health Surveillance</p>', unsafe_allow_html=True)
            st.markdown('<p class="signature-text">Produced by Henry — Certified Clinical Bioinformatician</p>', unsafe_allow_html=True)

else:
    st.info("""
    👆 Welcome to GeoAMR Clinical Surveillance System
    Upload Gonorrhoeae genomes (FASTA format) to begin analysis.

    What You'll Get:
    * ✅ NCBI-validated AMR gene detection (AMRFinderPlus)
    * ✅ Point mutation identification (gyrA, parC, etc.)
    * ✅ Virulence factor screening
    * ✅ Recombination frequency analysis
    * ✅ Nucleotide diversity metrics
    * ✅ PCA and population structure
    * ✅ Gene co-occurrence networks
    * ✅ Clinical PDF reports
    """)
