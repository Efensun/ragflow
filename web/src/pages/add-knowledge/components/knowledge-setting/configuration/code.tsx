import {
  AutoKeywordsItem,
  AutoQuestionsItem,
} from '@/components/auto-keywords-item';
import { DatasetConfigurationContainer } from '@/components/dataset-configuration-container';
import Delimiter from '@/components/delimiter';
import MaxTokenNumber from '@/components/max-token-number';
import ParseConfiguration from '@/components/parse-configuration';
import ContextRetrievalItems from '@/components/parse-configuration/context-retrieval-items';
import GraphRagItems from '@/components/parse-configuration/graph-rag-items';
import { Divider } from 'antd';
import { TagItems } from '../tag-item';
import { ChunkMethodItem, EmbeddingModelItem } from './common-item';

export function CodeConfiguration() {
  return (
    <section className="space-y-4 mb-4">
      <DatasetConfigurationContainer>
        <EmbeddingModelItem></EmbeddingModelItem>
        <ChunkMethodItem></ChunkMethodItem>
        <MaxTokenNumber max={2048}></MaxTokenNumber>
        <Delimiter></Delimiter>
      </DatasetConfigurationContainer>
      <Divider></Divider>
      <DatasetConfigurationContainer>
        <AutoKeywordsItem></AutoKeywordsItem>
        <AutoQuestionsItem></AutoQuestionsItem>
        <TagItems></TagItems>
      </DatasetConfigurationContainer>
      <Divider></Divider>
      <DatasetConfigurationContainer>
        <ParseConfiguration></ParseConfiguration>
      </DatasetConfigurationContainer>
      <Divider></Divider>
      <GraphRagItems></GraphRagItems>
      <Divider></Divider>
      <ContextRetrievalItems></ContextRetrievalItems>
    </section>
  );
}
