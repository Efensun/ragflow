import {
  AutoKeywordsItem,
  AutoQuestionsItem,
} from '@/components/auto-keywords-item';
import PageRank from '@/components/page-rank';
import ParseConfiguration from '@/components/parse-configuration';
import ContextRetrievalItems from '@/components/parse-configuration/context-retrieval-items';
import GraphRagItems from '@/components/parse-configuration/graph-rag-items';
import { TagItems } from '../tag-item';
import { ChunkMethodItem, EmbeddingModelItem } from './common-item';

export function EmailConfiguration() {
  return (
    <>
      <EmbeddingModelItem></EmbeddingModelItem>
      <ChunkMethodItem></ChunkMethodItem>

      <PageRank></PageRank>

      <>
        <AutoKeywordsItem></AutoKeywordsItem>
        <AutoQuestionsItem></AutoQuestionsItem>
      </>

      <ParseConfiguration></ParseConfiguration>

      <GraphRagItems marginBottom></GraphRagItems>
      
      <ContextRetrievalItems marginBottom></ContextRetrievalItems>

      <TagItems></TagItems>
    </>
  );
}
