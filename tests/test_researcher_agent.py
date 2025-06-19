"""
Tests for Researcher Agent functionality.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
import geopandas as gpd
import tempfile
import os

from tamagawa_to_z.researcher_agent.loader import ArtefactLoader, LoadedData
from tamagawa_to_z.researcher_agent.evaluator import Evaluator, IaEffect
from tamagawa_to_z.researcher_agent.rca import RootCauseAnalyzer
from tamagawa_to_z.researcher_agent.generator import ProposalGenerator, Proposal
from tamagawa_to_z.researcher_agent.scorer import ProposalScorer
from tamagawa_to_z.researcher_agent.formatter import MdFormatter, YamlFormatter
from tamagawa_to_z.researcher_agent.agent import ResearcherAgent


@pytest.fixture
def sample_loaded_data():
    """Create sample LoadedData for testing."""
    candidates = pd.DataFrame({
        'name': ['Rio Test', 'Lagoa Sample', 'Igarape Demo'],
        'geometry': ['POINT(-69 -8)', 'POINT(-69.1 -8.1)', 'POINT(-69.2 -8.2)'],
        'dist_km': [2.5, 3.0, 4.5],
        'occ_pct': [0.1, 0.05, 0.15]
    })
    
    known_sites = gpd.GeoDataFrame({
        'name': ['Site A', 'Site B'],
        'geometry': [None, None]  # Simplified for testing
    })
    
    return LoadedData(
        candidates=candidates,
        toponym_dict=pd.DataFrame(),
        known_sites=known_sites,
        param_yaml={'dist_threshold_km': 3.0, 'water_freq_threshold': 0.1},
        ia_report="Sample IA report content",
        ia_plan={'action': 'set_param', 'params': {'dist_threshold_km': 4.0}},
        code_snippets={'harmonizer': {'agent.py': {'process_toponyms': 'def process_toponyms():\n    pass'}}},
        meta_info={'run_id': 'test_123'}
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestArtefactLoader:
    """Test ArtefactLoader functionality."""
    
    def test_init(self, temp_dir):
        """Test ArtefactLoader initialization."""
        loader = ArtefactLoader(temp_dir)
        assert loader.artefact_dir == temp_dir
        assert loader.meta_info == {}
    
    def test_load_empty_directory(self, temp_dir):
        """Test loading from empty directory."""
        loader = ArtefactLoader(temp_dir)
        data = loader.load()
        
        assert isinstance(data, LoadedData)
        assert data.candidates.empty
        assert data.toponym_dict.empty
        assert data.ia_report == ""
        assert data.ia_plan == {}
    
    def test_load_with_candidates_csv(self, temp_dir):
        """Test loading with candidates CSV file."""
        # Create test CSV
        csv_file = temp_dir / "candidates.csv"
        test_df = pd.DataFrame({
            'name': ['Test Site'],
            'geometry': ['POINT(-69 -8)']
        })
        test_df.to_csv(csv_file, index=False)
        
        loader = ArtefactLoader(temp_dir)
        data = loader.load()
        
        assert not data.candidates.empty
        assert data.candidates.iloc[0]['name'] == 'Test Site'


class TestEvaluator:
    """Test Evaluator functionality."""
    
    def test_init(self, sample_loaded_data):
        """Test Evaluator initialization."""
        evaluator = Evaluator(sample_loaded_data)
        assert evaluator.data == sample_loaded_data
        assert 'workload' in evaluator.baseline_metrics
    
    def test_quick_simulate_no_plan(self, sample_loaded_data):
        """Test simulation with no IA plan."""
        sample_loaded_data.ia_plan = {}
        evaluator = Evaluator(sample_loaded_data)
        
        result = evaluator.quick_simulate()
        
        assert isinstance(result, IaEffect)
        assert result.confidence == 0.0
        assert "No IA proposal" in result.reasoning
    
    def test_quick_simulate_param_change(self, sample_loaded_data):
        """Test simulation with parameter change."""
        sample_loaded_data.ia_plan = {
            'action': 'set_param',
            'params': {'dist_threshold_km': 4.0}
        }
        evaluator = Evaluator(sample_loaded_data)
        
        result = evaluator.quick_simulate()
        
        assert isinstance(result, IaEffect)
        assert result.confidence > 0.0
        assert result.workload_change != 0


class TestRootCauseAnalyzer:
    """Test RootCauseAnalyzer functionality."""
    
    @patch('tamagawa_to_z.researcher_agent.rca.OpenAI')
    def test_init(self, mock_openai, sample_loaded_data):
        """Test RootCauseAnalyzer initialization."""
        mock_client = Mock()
        analyzer = RootCauseAnalyzer(mock_client, sample_loaded_data)
        
        assert analyzer.client == mock_client
        assert analyzer.data == sample_loaded_data
    
    @patch('tamagawa_to_z.researcher_agent.rca.OpenAI')
    def test_analyze_empty_data(self, mock_openai, sample_loaded_data):
        """Test analysis with minimal data."""
        sample_loaded_data.candidates = pd.DataFrame()
        
        mock_client = Mock()
        analyzer = RootCauseAnalyzer(mock_client, sample_loaded_data)
        
        result = analyzer.analyze()
        assert isinstance(result, list)


class TestProposalGenerator:
    """Test ProposalGenerator functionality."""
    
    @patch('tamagawa_to_z.researcher_agent.generator.OpenAI')
    def test_init(self, mock_openai, sample_loaded_data):
        """Test ProposalGenerator initialization."""
        mock_client = Mock()
        generator = ProposalGenerator(mock_client, sample_loaded_data, [], {})
        
        assert generator.client == mock_client
        assert generator.data == sample_loaded_data
    
    @patch('tamagawa_to_z.researcher_agent.generator.OpenAI')
    def test_generate_fallback_proposals(self, mock_openai, sample_loaded_data):
        """Test generation with fallback proposals."""
        mock_client = Mock()
        # Make LLM calls fail to trigger fallback
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        generator = ProposalGenerator(mock_client, sample_loaded_data, [], {})
        
        result = generator.generate(n=2)
        
        assert isinstance(result, list)
        assert len(result) <= 2
        for proposal in result:
            assert isinstance(proposal, Proposal)
            assert proposal.title
            assert proposal.changes


class TestProposalScorer:
    """Test ProposalScorer functionality."""
    
    def test_init_and_rank(self, sample_loaded_data):
        """Test ProposalScorer initialization and ranking."""
        proposals = [
            Proposal(
                id="A",
                title="Test Proposal A",
                changes=[{'action': 'set_param', 'params': {'test': 1}}],
                expected_effect={'recall@100': '+10%'},
                rationale="Test rationale",
                risk="Low risk",
                human_effort="★☆☆",
                priority="high"
            ),
            Proposal(
                id="B", 
                title="Test Proposal B",
                changes=[{'action': 'add_exclude_mask'}],
                expected_effect={'workload': '-100'},
                rationale="Test rationale B",
                risk="Medium risk",
                human_effort="★★☆", 
                priority="medium"
            )
        ]
        
        scorer = ProposalScorer(sample_loaded_data, proposals, {'improvement': 0.6, 'diversity': 0.2, 'cost': 0.2})
        
        result = scorer.rank()
        
        assert len(result) == 2
        assert all(hasattr(r, 'total_score') for r in result)
        assert all(hasattr(r, 'rank') for r in result)
        # Check that ranks are assigned properly
        assert result[0].rank == 1
        assert result[1].rank == 2


class TestFormatters:
    """Test formatter functionality."""
    
    def test_md_formatter(self, temp_dir, sample_loaded_data):
        """Test Markdown formatter."""
        formatter = MdFormatter(temp_dir)
        
        # Create dummy data
        from tamagawa_to_z.researcher_agent.scorer import RankedProposal
        proposal = Proposal(
            id="A",
            title="Test Proposal",
            changes=[{'action': 'set_param'}],
            expected_effect={'recall@100': '+5%'},
            rationale="Test",
            risk="Low",
            human_effort="★☆☆",
            priority="high"
        )
        
        ranked_proposals = [RankedProposal(
            proposal=proposal,
            improvement_score=0.8,
            diversity_score=0.7,
            cost_score=0.9,
            total_score=0.82,
            rank=1
        )]
        
        ia_eval = IaEffect(
            baseline_metrics={'workload': 500, 'recall@100': 0.3},
            estimated_metrics={'workload': 550, 'recall@100': 0.35},
            confidence=0.7,
            workload_change=50,
            reasoning="Test reasoning"
        )
        
        result_path = formatter.write_report(ranked_proposals, ia_eval, [])
        
        assert result_path.exists()
        assert result_path.suffix == '.md'
        
        content = result_path.read_text()
        assert "Test Proposal" in content
        assert "Test reasoning" in content
    
    def test_yaml_formatter(self, temp_dir):
        """Test YAML formatter."""
        formatter = YamlFormatter(temp_dir)
        
        # Create dummy data
        from tamagawa_to_z.researcher_agent.scorer import RankedProposal
        proposal = Proposal(
            id="A",
            title="Test Proposal",
            changes=[{'action': 'set_param', 'params': {'test': 1}}],
            expected_effect={'recall@100': '+5%'},
            rationale="Test",
            risk="Low",
            human_effort="★☆☆",
            priority="high"
        )
        
        ranked_proposals = [RankedProposal(
            proposal=proposal,
            improvement_score=0.8,
            diversity_score=0.7,
            cost_score=0.9,
            total_score=0.82,
            rank=1
        )]
        
        result_path = formatter.write_yaml(ranked_proposals)
        
        assert result_path.exists()
        assert result_path.suffix == '.yaml'
        
        # Validate YAML content
        import yaml
        with open(result_path, 'r') as f:
            data = yaml.safe_load(f)
        
        assert 'experiment_id' in data
        assert 'proposals' in data
        assert len(data['proposals']) == 1
        assert data['proposals'][0]['title'] == "Test Proposal"


class TestResearcherAgent:
    """Test main ResearcherAgent functionality."""
    
    @patch('tamagawa_to_z.researcher_agent.agent.OpenAI')
    def test_init(self, mock_openai):
        """Test ResearcherAgent initialization."""
        mock_client = Mock()
        config = {'weights': {'improvement': 0.6}}
        
        agent = ResearcherAgent(mock_client, config)
        
        assert agent.client == mock_client
        assert agent.cfg == config
    
    @patch('tamagawa_to_z.researcher_agent.agent.ArtefactLoader')
    @patch('tamagawa_to_z.researcher_agent.agent.OpenAI')
    def test_run_with_mocked_components(self, mock_openai, mock_loader_class, temp_dir, sample_loaded_data):
        """Test full run with mocked components."""
        # Setup mocks
        mock_client = Mock()
        mock_loader = Mock()
        mock_loader.load.return_value = sample_loaded_data
        mock_loader_class.return_value = mock_loader
        
        config = {
            'weights': {'improvement': 0.6, 'diversity': 0.2, 'cost': 0.2}
        }
        
        agent = ResearcherAgent(mock_client, config)
        
        # Mock LLM responses to avoid API calls
        with patch.multiple(
            'tamagawa_to_z.researcher_agent.rca.RootCauseAnalyzer',
            analyze=Mock(return_value=[])
        ), patch.multiple(
            'tamagawa_to_z.researcher_agent.generator.ProposalGenerator',
            generate=Mock(return_value=[])
        ):
            
            result = agent.run(temp_dir, temp_dir)
            
            assert len(result) == 2  # (report_path, yaml_path)
            report_path, yaml_path = result
            
            assert isinstance(report_path, Path)
            assert isinstance(yaml_path, Path)


if __name__ == "__main__":
    # Run tests if script is called directly
    pytest.main([__file__])